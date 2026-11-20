//! The Nielsen classes `Ni(C, 23A, 23B)` of M23 for a rational class `C`, complex conjugation
//! on them, real structures and the number of components of the real locus (ovals) (WP-P,
//! HANDOFF 4.2 step 7 and 4.3).
//!
//! MAGMA: ch. 58 (permutation groups: conjugation, `IsConjugate`); digest
//! `rustmath-lava/docs/Chapter_58_Permutation_Groups.md`.
//!
//! Sources:
//! - HJLPPZ (arXiv:2608.08538v2), 3.1: the seven classes of `(2A, 23A, 23B)` and complex
//!   conjugation `(g1, g2, g3) -> (g1^-1, g3^-1, g2^-1)` for branch points `0, s, conj(s)`.
//! - The owner's paper (`johnjanik/M23` at `2d11038`, `paper/golay_m23.tex`), the theorem on the
//!   real structure (ii) and Section 7, and `m23golay/nielsen.py` (`canonical`, `index_of`,
//!   `real_structure`, `real_components_general`), ported as algorithms.
//!
//! A class is stored by the canonical first entry of its triples with `g2` the published
//! `g2`: the lexicographically least `<g2>`-conjugate `g2^-k a g2^k` (as a 0-based image list,
//! the order of Python tuples). The classes are numbered by their sorted canonical entries,
//! which is the owner's numbering (`passports_C_23A_23B.json`, `group_facts.json`).
//!
//! Every triple of `(C, 23A, 23B)` with `C` rational and not 1A generates M23: it contains a
//! 23-cycle, so it is transitive, and an element of order 2 to 8 (generation theorem of
//! `three_point.rs`). `C(g2) = <g2>` acts freely on the candidates.

use std::fmt;

use super::classes::{ClassIdentifier, ClassLists, M23Class};
use super::group::{
    conjugate, cycles, fixed_mask, identity23, inverse, rmul, M23Error, Perm23, M23,
};

/// A triple `(g1, g2, g3)` with `g1 g2 g3 = 1` (right action).
pub type Triple = (Perm23, Perm23, Perm23);

/// Errors of the Nielsen-class layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NielsenError {
    /// `g1 g2 g3 != 1`.
    ProductNotOne,
    /// No element of M23 conjugates the second entry to the published `g2`.
    SecondEntryNotIn23A,
    /// The normalized first entry is not in the class.
    NotInClass,
    /// The list of the class was not built.
    ClassListMissing(M23Class),
    /// `C` must be a rational class other than 1A.
    NotARationalClass(M23Class),
    /// A class fixed by complex conjugation without a real structure (impossible).
    RealStructureMissing(usize),
    /// The oval graph is malformed, or `c0` is not a real structure of the triple.
    OvalGraph(String),
    /// An error of the group layer, or a failed internal check.
    Group(M23Error),
}

impl fmt::Display for NielsenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NielsenError::ProductNotOne => write!(f, "the product of the triple is not 1"),
            NielsenError::SecondEntryNotIn23A => write!(f, "the second entry is not in 23A"),
            NielsenError::NotInClass => write!(f, "the triple is not in the Nielsen class"),
            NielsenError::ClassListMissing(c) => write!(f, "the list of class {c} was not built"),
            NielsenError::NotARationalClass(c) => {
                write!(f, "{c} is not a rational class other than 1A")
            }
            NielsenError::RealStructureMissing(i) => {
                write!(f, "class {i} is fixed but has no real structure")
            }
            NielsenError::OvalGraph(s) => write!(f, "oval graph: {s}"),
            NielsenError::Group(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for NielsenError {}

impl From<M23Error> for NielsenError {
    fn from(e: M23Error) -> NielsenError {
        NielsenError::Group(e)
    }
}

/// The least `<g2>`-conjugate of `a` (lexicographic on image lists).
fn canonical(a: &Perm23, g2: &Perm23) -> Perm23 {
    let mut best = *a;
    let mut cur = *a;
    for _ in 0..22 {
        cur = conjugate(&cur, g2);
        if cur < best {
            best = cur;
        }
    }
    best
}

/// An `h` in M23 with `h^-1 a h = b`, for 23-cycles `a`, `b`: the 23 maps sending the cycle
/// of `a` to that of `b`, each sifted. `None` if none lies in M23 (or `a` is not a 23-cycle).
fn conjugator_23(g: &M23, a: &Perm23, b: &Perm23) -> Option<Perm23> {
    for start in 0..23 {
        let mut h = [0u8; 23];
        let (mut x, mut y) = (0usize, start);
        for _ in 0..23 {
            h[x] = y as u8;
            x = a[x] as usize;
            y = b[y] as usize;
        }
        if g.contains(&h) {
            return Some(h);
        }
    }
    None
}

/// The complex conjugate triple `(g1^-1, g3^-1, g2^-1)` (HJLPPZ 3.1).
pub fn complex_conjugate(t: &Triple) -> Triple {
    (inverse(&t.0), inverse(&t.2), inverse(&t.1))
}

/// `Ni(C, 23A, 23B)` with `g2` normalized to the published `g2`.
pub struct NielsenClass23<'a> {
    g: &'a M23,
    class: M23Class,
    reps: Vec<Perm23>,
    candidates: usize,
}

impl<'a> NielsenClass23<'a> {
    /// The classes of `(c, 23A, 23B)`: the `a` in `c` with `a g2` in 23A, up to conjugation by
    /// `<g2>`. The candidates come from `lists.get(c)` (`Err(ClassListMissing)` without it), so
    /// a list of elements outside the class cannot be passed. Checked: the number of candidates
    /// is 23 times the number of classes (`C(g2) = <g2>` acts freely).
    pub fn new(
        g: &'a M23,
        id: &ClassIdentifier,
        lists: &ClassLists,
        c: M23Class,
    ) -> Result<NielsenClass23<'a>, NielsenError> {
        if !c.is_rational() || c == M23Class::C1A {
            return Err(NielsenError::NotARationalClass(c));
        }
        let list = lists.get(c).ok_or(NielsenError::ClassListMissing(c))?;
        let g2 = g.g2();
        let mut good = Vec::new();
        for a in list {
            let p = rmul(a, &g2);
            if super::group::num_cycles(&p) == 1 && id.class_of_member(&p)? == M23Class::C23A {
                good.push(*a);
            }
        }
        let mut reps: Vec<Perm23> = good.iter().map(|a| canonical(a, &g2)).collect();
        reps.sort_unstable();
        reps.dedup();
        if reps.len() * 23 != good.len() {
            return Err(NielsenError::Group(M23Error::Construction(format!(
                "C(g2) does not act freely: {} candidates, {} classes",
                good.len(),
                reps.len()
            ))));
        }
        Ok(NielsenClass23 {
            g,
            class: c,
            reps,
            candidates: good.len(),
        })
    }

    /// The class `C`.
    pub fn class(&self) -> M23Class {
        self.class
    }

    /// The number of Nielsen classes.
    pub fn len(&self) -> usize {
        self.reps.len()
    }

    /// Whether there is no class.
    pub fn is_empty(&self) -> bool {
        self.reps.is_empty()
    }

    /// The number of `a` in `C` with `a g2` in 23A (23 times [`NielsenClass23::len`]).
    pub fn num_candidates(&self) -> usize {
        self.candidates
    }

    /// The sorted canonical first entries (the owner's `representatives_g1`).
    pub fn representatives(&self) -> &[Perm23] {
        &self.reps
    }

    /// The triple `(a, g2, (a g2)^-1)` of class `i`. Panics if `i >= len()`.
    pub fn triple(&self, i: usize) -> Triple {
        let a = self.reps[i];
        let g2 = self.g.g2();
        (a, g2, inverse(&rmul(&a, &g2)))
    }

    /// The index of the class of `t`. `Err(ProductNotOne)`, `Err(SecondEntryNotIn23A)` when no
    /// `h` in M23 has `h^-1 t.1 h = g2`, `Err(NotInClass)` when the normalized first entry is
    /// not a representative. An `Ok` implies that the triple lies in M23 and in the class.
    pub fn index_of(&self, t: &Triple) -> Result<usize, NielsenError> {
        let (g1, g2, g3) = t;
        if rmul(&rmul(g1, g2), g3) != identity23() {
            return Err(NielsenError::ProductNotOne);
        }
        let h = conjugator_23(self.g, g2, &self.g.g2()).ok_or(NielsenError::SecondEntryNotIn23A)?;
        let a = canonical(&conjugate(g1, &h), &self.g.g2());
        self.reps
            .binary_search(&a)
            .map_err(|_| NielsenError::NotInClass)
    }

    /// Complex conjugation on the classes: `i -> index_of(g1^-1, g3^-1, g2^-1)`.
    pub fn complex_conjugation(&self) -> Result<Vec<usize>, NielsenError> {
        (0..self.len())
            .map(|i| self.index_of(&complex_conjugate(&self.triple(i))))
            .collect()
    }

    /// The real structure of class `i`: `None` if the class is not fixed by complex
    /// conjugation; otherwise the involution `c0` in M23 with `c0 g1 c0 = g1^-1` and
    /// `c0 g2 c0 = g3^-1`, found as `g2^k h` with `h^-1 g2 h = g3^-1` (owner's
    /// `real_structure`). A fixed class without one is `Err(RealStructureMissing)`. Panics if
    /// `i >= len()`.
    pub fn real_structure(&self, i: usize) -> Result<Option<Perm23>, NielsenError> {
        let t = self.triple(i);
        if self.index_of(&complex_conjugate(&t))? != i {
            return Ok(None);
        }
        let (g1, g2, g3) = t;
        let g3i = inverse(&g3);
        let h = conjugator_23(self.g, &g2, &g3i).ok_or(NielsenError::RealStructureMissing(i))?;
        let g1i = inverse(&g1);
        let mut pk = identity23();
        for _ in 0..23 {
            let c0 = rmul(&pk, &h);
            if conjugate(&g1, &c0) == g1i {
                if conjugate(&g2, &c0) != g3i || rmul(&c0, &c0) != identity23() {
                    return Err(NielsenError::RealStructureMissing(i));
                }
                return Ok(Some(c0));
            }
            pk = rmul(&pk, &g2);
        }
        Err(NielsenError::RealStructureMissing(i))
    }

    /// `|Fix(c0) cap Fix(g1)|` for class `i`: the real points above 0. Panics if `i >= len()`.
    pub fn real_points_above_0(&self, i: usize, c0: &Perm23) -> usize {
        (fixed_mask(c0) & fixed_mask(&self.triple(i).0)).count_ones() as usize
    }

    /// The number of components of `X(R)` for class `i` with real structure `c0` (owner's
    /// `real_components_general`): vertices `A = Fix(c0)` and `B = Fix(c0 g1)`; each `x` in `A`
    /// is joined to `x^(g2^-1)` in `B`; on each cycle of `g1` with points of `A u B`, the
    /// pattern `(1, 1)` (odd length) joins the `A` point to the `B` point, and `(2, 0)` or
    /// `(0, 2)` (even length) joins the two points. Every vertex must have degree 2; the
    /// components are the cycles of the graph. `Err(OvalGraph)` otherwise, or if `c0` is not
    /// a real structure of the triple. Panics if `i >= len()`.
    pub fn real_components(&self, i: usize, c0: &Perm23) -> Result<usize, NielsenError> {
        let (g1, g2, g3) = self.triple(i);
        if conjugate(&g1, c0) != inverse(&g1)
            || conjugate(&g2, c0) != inverse(&g3)
            || rmul(c0, c0) != identity23()
        {
            return Err(NielsenError::OvalGraph("c0 is not a real structure".into()));
        }
        let c0g1 = rmul(c0, &g1);
        let a = fixed_mask(c0);
        let b = fixed_mask(&c0g1);
        let in_a = |x: usize| (a >> x) & 1 == 1;
        let in_b = |x: usize| (b >> x) & 1 == 1;
        // vertices: A point x -> x, B point x -> 23 + x
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); 46];
        let edge = |u: usize, v: usize, adj: &mut Vec<Vec<usize>>| {
            adj[u].push(v);
            adj[v].push(u);
        };
        let m = inverse(&g2);
        for x in (0..23).filter(|&x| in_a(x)) {
            let y = m[x] as usize;
            if !in_b(y) {
                return Err(NielsenError::OvalGraph(format!(
                    "x^(g2^-1) of the A point {x} is not in B"
                )));
            }
            edge(x, 23 + y, &mut adj);
        }
        for cyc in cycles(&g1) {
            let ca: Vec<usize> = cyc
                .iter()
                .map(|&x| x as usize)
                .filter(|&x| in_a(x))
                .collect();
            let cb: Vec<usize> = cyc
                .iter()
                .map(|&x| x as usize)
                .filter(|&x| in_b(x))
                .collect();
            match (ca.len(), cb.len()) {
                (1, 1) if cyc.len() % 2 == 1 => edge(ca[0], 23 + cb[0], &mut adj),
                (2, 0) if cyc.len() % 2 == 0 => edge(ca[0], ca[1], &mut adj),
                (0, 2) if cyc.len() % 2 == 0 => edge(23 + cb[0], 23 + cb[1], &mut adj),
                (0, 0) => {}
                other => {
                    return Err(NielsenError::OvalGraph(format!(
                        "real points {other:?} on a cycle of length {}",
                        cyc.len()
                    )))
                }
            }
        }
        let verts: Vec<usize> = (0..23)
            .filter(|&x| in_a(x))
            .chain((0..23).filter(|&x| in_b(x)).map(|x| 23 + x))
            .collect();
        if let Some(&v) = verts.iter().find(|&&v| adj[v].len() != 2) {
            return Err(NielsenError::OvalGraph(format!(
                "vertex {v} has degree {}",
                adj[v].len()
            )));
        }
        let mut seen = [false; 46];
        let mut comps = 0;
        for &v in &verts {
            if seen[v] {
                continue;
            }
            comps += 1;
            let mut st = vec![v];
            while let Some(u) = st.pop() {
                if !seen[u] {
                    seen[u] = true;
                    st.extend(adj[u].iter().copied());
                }
            }
        }
        Ok(comps)
    }
}
