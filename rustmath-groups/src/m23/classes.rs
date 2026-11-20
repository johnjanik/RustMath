//! The 17 conjugacy classes of M23, with the split pairs identified exactly, class sizes and
//! centralizer orders.
//!
//! MAGMA: ch. 58 (`Classes`, `ClassMap`, `Centralizer`, `IsConjugate`); digest
//! `rustmath-lava/docs/Chapter_58_Permutation_Groups.md`. MAGMA computes classes by random
//! search and Leon's backtrack; here M23 is small enough for exact invariants.
//!
//! Mathematics and sources:
//! - ATLAS of Finite Groups (Conway, Curtis, Norton, Parker, Wilson, 1985), M23, p. 71: the 17
//!   classes, their cycle types and centralizer orders.
//! - The owner's M23 repository (`johnjanik/M23` at `2d11038`): `scripts/passports_all.py`
//!   (the split invariants of step 3), `m23golay/m23.py` (`SIGNATURES`, `sign_of_23_element`),
//!   ported as algorithms. HANDOFF Section 4.2, step 3.
//! - Le Lay, "Arithmetic and geometry of an M23 Hurwitz scheme", Section 5.4: a 23-cycle of M23
//!   preserves exactly two Steiner systems S(4,7,23), the two cyclic ones.
//!
//! The invariants (all on right-action image lists):
//! - cycle type, from the signature `(Fix g, Fix g^2, Fix g^3)`, which separates the 12 types;
//! - the Golay trace `tr(g) = sum_k v_k[g^-1(i_k)]` over `F_2` (7AB, 14AB, 15AB, 23AB), with the
//!   reduced echelon basis `(v_k, i_k)` of the even Golay code;
//! - the heptad-position family (11AB): label each 11-cycle of `g` by `Z/11` along `g` and
//!   collect the position sets of the heptads that meet it in at least 3 points;
//! - the cyclic Steiner sign (23AB, a second invariant): label the points by `0^(g^i) -> i`; the
//!   transported system is one of the two cyclic systems of `rustmath_coding::cyclic_golay`.
//!
//! Why the identification is exact. For a split type with representative `x`: (1) each
//! invariant is a class function of M23 (the trace is the trace of an `F_2[M23]`-module, and
//! the family and the sign are built from the Steiner system, which M23 preserves); (2) it takes
//! different values on `x` and `x^-1` (checked in [`ClassIdentifier::new`]); (3) the number of
//! elements of the type with the value of `x` is `|G| / |C_G(x)|` (the full count is
//! [`census`], checked against GAP in the release test `full_census`). So the two classes of the
//! type are exactly the two level sets.
//!
//! Naming (a convention inside each pair, HANDOFF Section 1): 23A is the class of the published
//! `g2`; 11A the class of `h11: 0^(g2^i) -> 0^(g2^(2i))`, an element of `N(<g2>) = 23:11`; 7A and
//! 15A the classes of Golay trace 1; 14A the class whose squares lie in 7A. Over `F_2`,
//! `tr(x^2) = tr(x)^2 = tr(x)`, so 14A also has trace 1 (design review 2), and squaring
//! preserves each trace class, as GAP's power map 2 does on 7AB, 15AB and 23AB. With these
//! names the classes match GAP's `CharacterTable("M23")` columns up to a table automorphism,
//! under which every structure constant is invariant (`atlas.rs`).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use rustmath_coding::cyclic_golay::{self, GolaySign};

use super::group::{conjugate, inverse, mask_image, order, rmul, M23Error, Perm23, M23, M23_ORDER};
use crate::transitive23::identity;

/// The 17 conjugacy classes of M23, in ATLAS order (the column order of GAP's
/// `CharacterTable("M23")`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum M23Class {
    /// The identity.
    C1A,
    /// Involutions, `1^7 2^8`.
    C2A,
    /// `1^5 3^6`.
    C3A,
    /// `1^3 2^2 4^4`.
    C4A,
    /// `1^3 5^4`.
    C5A,
    /// `1 2^2 3^2 6^2`.
    C6A,
    /// `1^2 7^3`, Golay trace 1.
    C7A,
    /// `1^2 7^3`, Golay trace 0.
    C7B,
    /// `1 2 4 8^2`.
    C8A,
    /// `1 11^2`, the class of `h11` (see the module docs).
    C11A,
    /// `1 11^2`, the inverses of 11A.
    C11B,
    /// `2 7 14`, squares in 7A.
    C14A,
    /// `2 7 14`, squares in 7B.
    C14B,
    /// `3 5 15`, Golay trace 1.
    C15A,
    /// `3 5 15`, Golay trace 0.
    C15B,
    /// 23-cycles, the class of the published `g2`.
    C23A,
    /// 23-cycles, the class of the published `g3`.
    C23B,
}

const NAMES: [&str; 17] = [
    "1A", "2A", "3A", "4A", "5A", "6A", "7A", "7B", "8A", "11A", "11B", "14A", "14B", "15A", "15B",
    "23A", "23B",
];
const ORDERS: [u32; 17] = [1, 2, 3, 4, 5, 6, 7, 7, 8, 11, 11, 14, 14, 15, 15, 23, 23];
/// The cycle type (index 0..12) of each class.
const CLASS_TYPE: [usize; 17] = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11];
/// The classes of each of the 12 cycle types.
const TYPE_CLASSES: [&[usize]; 12] = [
    &[0],
    &[1],
    &[2],
    &[3],
    &[4],
    &[5],
    &[6, 7],
    &[8],
    &[9, 10],
    &[11, 12],
    &[13, 14],
    &[15, 16],
];
const T7: usize = 6;
const T11: usize = 8;
const T14: usize = 9;
const T15: usize = 10;
const T23: usize = 11;

const CT_1A: [u8; 23] = [1; 23];
const CT_2A: [u8; 15] = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2];
const CT_3A: [u8; 11] = [1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3];
const CT_4A: [u8; 9] = [1, 1, 1, 2, 2, 4, 4, 4, 4];
const CT_5A: [u8; 7] = [1, 1, 1, 5, 5, 5, 5];
const CT_6A: [u8; 7] = [1, 2, 2, 3, 3, 6, 6];
const CT_7: [u8; 5] = [1, 1, 7, 7, 7];
const CT_8A: [u8; 5] = [1, 2, 4, 8, 8];
const CT_11: [u8; 3] = [1, 11, 11];
const CT_14: [u8; 3] = [2, 7, 14];
const CT_15: [u8; 3] = [3, 5, 15];
const CT_23: [u8; 1] = [23];

impl M23Class {
    /// All 17 classes in ATLAS order.
    pub const ALL: [M23Class; 17] = [
        M23Class::C1A,
        M23Class::C2A,
        M23Class::C3A,
        M23Class::C4A,
        M23Class::C5A,
        M23Class::C6A,
        M23Class::C7A,
        M23Class::C7B,
        M23Class::C8A,
        M23Class::C11A,
        M23Class::C11B,
        M23Class::C14A,
        M23Class::C14B,
        M23Class::C15A,
        M23Class::C15B,
        M23Class::C23A,
        M23Class::C23B,
    ];

    /// The position in [`M23Class::ALL`] (0-based GAP column).
    pub fn index(self) -> usize {
        self as usize
    }

    /// The class of a position in [`M23Class::ALL`].
    pub fn from_index(i: usize) -> Option<M23Class> {
        M23Class::ALL.get(i).copied()
    }

    /// The ATLAS name, for example `"23A"`.
    pub fn name(self) -> &'static str {
        NAMES[self.index()]
    }

    /// The class of an ATLAS name, upper or lower case (`"23A"`, `"23a"`).
    pub fn from_name(s: &str) -> Option<M23Class> {
        let up = s.trim().to_ascii_uppercase();
        NAMES
            .iter()
            .position(|&n| n == up)
            .map(|i| M23Class::ALL[i])
    }

    /// The order of the elements of the class.
    pub fn element_order(self) -> u32 {
        ORDERS[self.index()]
    }

    /// The cycle type on the 23 points: the cycle lengths in ascending order, fixed points
    /// included (as `transitive23::cycle_type`).
    pub fn cycle_type(self) -> &'static [u8] {
        match CLASS_TYPE[self.index()] {
            0 => &CT_1A,
            1 => &CT_2A,
            2 => &CT_3A,
            3 => &CT_4A,
            4 => &CT_5A,
            5 => &CT_6A,
            6 => &CT_7,
            7 => &CT_8A,
            8 => &CT_11,
            9 => &CT_14,
            10 => &CT_15,
            _ => &CT_23,
        }
    }

    /// The number of cycles on the 23 points (fixed points included).
    pub fn num_cycles(self) -> usize {
        self.cycle_type().len()
    }

    /// The class of the inverses. The split pairs are exchanged (ATLAS power map `-1`).
    pub fn inverse(self) -> M23Class {
        let cl = TYPE_CLASSES[CLASS_TYPE[self.index()]];
        if cl.len() == 1 {
            self
        } else if cl[0] == self.index() {
            M23Class::ALL[cl[1]]
        } else {
            M23Class::ALL[cl[0]]
        }
    }

    /// Whether the class is rational (every character is rational on it): 1A, 2A, 3A, 4A, 5A,
    /// 6A, 8A. Exactly the classes that are not split.
    pub fn is_rational(self) -> bool {
        TYPE_CLASSES[CLASS_TYPE[self.index()]].len() == 1
    }

    /// The other class of the same cycle type, for a split class.
    pub fn partner(self) -> Option<M23Class> {
        if self.is_rational() {
            None
        } else {
            Some(self.inverse())
        }
    }
}

impl std::fmt::Display for M23Class {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `(Fix g, Fix g^2, Fix g^3)`.
#[inline]
pub fn signature(p: &Perm23) -> (u8, u8, u8) {
    let (mut f1, mut f2, mut f3) = (0u8, 0u8, 0u8);
    for (i, &a) in p.iter().enumerate() {
        let b = p[a as usize];
        let c = p[b as usize];
        f1 += (a as usize == i) as u8;
        f2 += (b as usize == i) as u8;
        f3 += (c as usize == i) as u8;
    }
    (f1, f2, f3)
}

/// The cycle type (0..12, in the order of `m23.py` `SIGNATURES`) of a signature, or `None` for
/// a signature outside the 12 types of M23.
#[inline]
pub(crate) fn type_of_signature(s: (u8, u8, u8)) -> Option<usize> {
    Some(match s {
        (23, 23, 23) => 0,
        (7, 23, 7) => 1,
        (5, 5, 23) => 2,
        (3, 7, 3) => 3,
        (3, 3, 3) => 4,
        (1, 5, 7) => 5,
        (2, 2, 2) => 6,
        (1, 3, 1) => 7,
        (1, 1, 1) => 8,
        (0, 2, 0) => 9,
        (0, 0, 3) => 10,
        (0, 0, 0) => 11,
        _ => return None,
    })
}

/// The cycle type (0..12) of `p`, if it is one of the 12 types of M23.
#[inline]
pub(crate) fn type_of(p: &Perm23) -> Option<usize> {
    type_of_signature(signature(p))
}

/// The cycle type (0..12) of a class.
#[inline]
pub(crate) fn class_type(c: M23Class) -> usize {
    CLASS_TYPE[c.index()]
}

/// Default budget of the `C_Sym(x)` search of [`ClassIdentifier::centralizer_order`].
pub const SIFT_BUDGET: u64 = 1_000_000;
/// Default cap of the conjugation orbit of [`ClassIdentifier::centralizer_order`].
pub const ORBIT_CAP: usize = 100_000;

fn rot11(s: u16, k: u32) -> u16 {
    if k == 0 {
        return s;
    }
    ((((s as u32) << k) | ((s as u32) >> (11 - k))) & 0x7ff) as u16
}

/// Identifies the class of an element of M23 (conventions in the module docs).
pub struct ClassIdentifier {
    heptads: Vec<u32>,
    golay: Vec<(u32, u8)>,
    /// For each split type: the value of `tr(x^-1)` of the A class.
    tr_inv_a: [u8; 12],
    /// The heptad-position families of 11A and 11B.
    fam11: [Vec<(u16, u32)>; 2],
    /// The least rotation of each subset of `Z/11`.
    canon11: Vec<u16>,
    /// Translation classes of position sets that occur for 11A and not for 11B.
    a_only11: Vec<bool>,
    reps: [Perm23; 17],
    centralizers: [OnceLock<u64>; 17],
}

impl std::fmt::Debug for ClassIdentifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClassIdentifier")
            .field("tr_inv_a", &self.tr_inv_a)
            .finish_non_exhaustive()
    }
}

/// One representative of each of the 12 cycle types, from a deterministic walk in `<g1, g2>`
/// (a linear congruential generator chooses the next generator). Every element found lies in
/// M23 (a product of generators).
fn type_representatives(g1: &Perm23, g2: &Perm23) -> Result<[Perm23; 12], M23Error> {
    let mut reps: [Option<Perm23>; 12] = [None; 12];
    reps[0] = Some(identity());
    let mut p = identity();
    let mut state: u64 = 0x2323_2323;
    let mut steps = 0u32;
    while reps.iter().any(|r| r.is_none()) {
        steps += 1;
        if steps > 1_000_000 {
            return Err(M23Error::Construction(
                "the walk did not find all 12 cycle types".into(),
            ));
        }
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        p = rmul(&p, if state >> 63 == 1 { g1 } else { g2 });
        let o = order(&p);
        let mut q = p;
        for _ in 1..o {
            let t = type_of(&q).ok_or_else(|| {
                M23Error::Construction("a product of the generators has a foreign type".into())
            })?;
            if reps[t].is_none() {
                reps[t] = Some(q);
            }
            q = rmul(&q, &p);
        }
    }
    Ok(reps.map(|r| r.expect("all found")))
}

impl ClassIdentifier {
    /// Builds the invariants and fixes the names of the split classes (module docs). Checked:
    /// each split invariant takes different values on the representative and its inverse;
    /// `h11` lies in M23 and has order 11; the cyclic sign separates `g2` and `g3`; some
    /// translation class of position sets occurs only for 11A. A failure is
    /// `Err(Construction)`, a defect.
    pub fn new(g: &M23) -> Result<ClassIdentifier, M23Error> {
        let (g1, g2) = (g.g1(), g.g2());
        let canon11: Vec<u16> = (0u16..2048)
            .map(|s| (0..11).map(|k| rot11(s, k)).min().expect("11 rotations"))
            .collect();
        let mut id = ClassIdentifier {
            heptads: g.heptads().to_vec(),
            golay: g.even_golay_basis().to_vec(),
            tr_inv_a: [0; 12],
            fam11: [Vec::new(), Vec::new()],
            canon11,
            a_only11: vec![false; 2048],
            reps: [identity(); 17],
            centralizers: std::array::from_fn(|_| OnceLock::new()),
        };
        let treps = type_representatives(&g1, &g2)?;
        let fail = |s: &str| Err(M23Error::Construction(s.to_string()));

        // 23A: the class of the published g2.
        if id.trace_of_inverse(&g2) == id.trace_of_inverse(&inverse(&g2)) {
            return fail("the Golay trace does not separate g2 and g2^-1");
        }
        id.tr_inv_a[T23] = id.trace_of_inverse(&g2);
        match (id.steiner_sign_raw(&g2)?, id.steiner_sign_raw(&g.g3())?) {
            (Some(a), Some(b)) if a != b => {}
            _ => return fail("the cyclic sign does not separate g2 and g3"),
        }

        // 11A: the class of h11 with h11(0^(g2^i)) = 0^(g2^(2i)), in N(<g2>) = 23:11.
        let mut orbit = [0u8; 23];
        let mut x = 0usize;
        for o in orbit.iter_mut() {
            *o = x as u8;
            x = g2[x] as usize;
        }
        let mut h11 = [0u8; 23];
        for i in 0..23 {
            h11[orbit[i] as usize] = orbit[(2 * i) % 23];
        }
        if !g.contains(&h11) || order(&h11) != 11 || type_of(&h11) != Some(T11) {
            return fail("h11 is not an element of order 11 of M23");
        }
        id.fam11 = [id.family_raw(&h11), id.family_raw(&inverse(&h11))];
        if id.fam11[0] == id.fam11[1] {
            return fail("the heptad family does not separate 11A and 11B");
        }
        let sa: HashSet<u16> = id.fam11[0]
            .iter()
            .map(|&(s, _)| id.canon11[s as usize])
            .collect();
        let sb: HashSet<u16> = id.fam11[1]
            .iter()
            .map(|&(s, _)| id.canon11[s as usize])
            .collect();
        let mut k = 0;
        for &c in sa.difference(&sb) {
            id.a_only11[c as usize] = true;
            k += 1;
        }
        if k == 0 {
            return fail("no translation class of position sets occurs only for 11A");
        }

        // 7A, 15A: Golay trace 1.
        for t in [T7, T15] {
            let rep = treps[t];
            let (a, b) = (id.golay_trace(&rep), id.golay_trace(&inverse(&rep)));
            if a == b {
                return fail("the Golay trace does not separate 7AB or 15AB");
            }
            let a_rep = if a == 1 { rep } else { inverse(&rep) };
            id.tr_inv_a[t] = id.trace_of_inverse(&a_rep);
        }
        // 14A: the class whose squares lie in 7A.
        {
            let rep = treps[T14];
            if id.golay_trace(&rep) == id.golay_trace(&inverse(&rep)) {
                return fail("the Golay trace does not separate 14AB");
            }
            let sq = rmul(&rep, &rep);
            let a_rep = if id.class_of_member(&sq)? == M23Class::C7A {
                rep
            } else {
                inverse(&rep)
            };
            id.tr_inv_a[T14] = id.trace_of_inverse(&a_rep);
        }
        // One representative per class.
        for t in 0..12 {
            let r = if t == T23 {
                g2
            } else if t == T11 {
                h11
            } else {
                treps[t]
            };
            let cl = TYPE_CLASSES[t];
            if cl.len() == 1 {
                id.reps[cl[0]] = r;
            } else {
                let c = id.class_of_member(&r)?.index();
                let other = if c == cl[0] { cl[1] } else { cl[0] };
                id.reps[c] = r;
                id.reps[other] = inverse(&r);
            }
        }
        Ok(id)
    }

    /// `tr(g^-1 | G0) = sum_k v_k[g(i_k)]` (no inversion needed).
    #[inline]
    fn trace_of_inverse(&self, g: &Perm23) -> u8 {
        let mut t = 0u32;
        for &(v, piv) in &self.golay {
            t ^= (v >> g[piv as usize]) & 1;
        }
        t as u8
    }

    /// The Golay trace `tr(g | G0) = sum_k v_k[g^-1(i_k)]` over `F_2` (HANDOFF 4.2 step 3).
    pub fn golay_trace(&self, p: &Perm23) -> u8 {
        self.trace_of_inverse(&inverse(p))
    }

    /// The family of position sets of an element with two 11-cycles (unchecked).
    fn family_raw(&self, x: &Perm23) -> Vec<(u16, u32)> {
        let mut seen = [false; 23];
        let mut pos = [0u8; 23];
        let mut cyc_masks = Vec::new();
        for i in 0..23 {
            if !seen[i] {
                let mut j = i;
                let mut k = 0u8;
                let mut m = 0u32;
                while !seen[j] {
                    seen[j] = true;
                    pos[j] = k;
                    m |= 1 << j;
                    k += 1;
                    j = x[j] as usize;
                }
                if k == 11 {
                    cyc_masks.push(m);
                }
            }
        }
        let mut fam: HashMap<u16, u32> = HashMap::new();
        for &h in &self.heptads {
            for &cm in &cyc_masks {
                let inter = h & cm;
                if inter.count_ones() >= 3 {
                    let mut s = 0u16;
                    let mut v = inter;
                    while v != 0 {
                        s |= 1 << pos[v.trailing_zeros() as usize];
                        v &= v - 1;
                    }
                    *fam.entry(s).or_insert(0) += 1;
                }
            }
        }
        let mut f: Vec<(u16, u32)> = fam.into_iter().collect();
        f.sort_unstable();
        f
    }

    /// The heptad-position family of an element with two 11-cycles: the multiset of the
    /// position sets (subsets of `Z/11` as 11-bit masks, with multiplicities), sorted.
    /// `Err(InvariantFailed)` if `p` does not have two 11-cycles, or if its family is neither
    /// that of 11A nor that of 11B (impossible in M23).
    pub fn heptad_family_11(&self, p: &Perm23) -> Result<Vec<(u16, u32)>, M23Error> {
        let mut lens: Vec<usize> = super::group::cycles(p).iter().map(|c| c.len()).collect();
        lens.sort_unstable();
        if lens != [1, 11, 11] {
            return Err(M23Error::InvariantFailed(
                "heptad_family_11 needs an element of cycle type 1 11^2".into(),
            ));
        }
        let f = self.family_raw(p);
        if f == self.fam11[0] || f == self.fam11[1] {
            Ok(f)
        } else {
            Err(M23Error::InvariantFailed(
                "the heptad family is neither that of 11A nor that of 11B".into(),
            ))
        }
    }

    /// The fast exact 11 test: some position set lies in a translation class that occurs for
    /// 11A and not for 11B. A function of the family, so exact on M23 (module docs).
    #[inline]
    fn is_11a(&self, x: &Perm23) -> bool {
        let mut seen = 0u32;
        let mut pos = [0u8; 23];
        let mut cm = [0u32; 2];
        let mut nc = 0;
        for i in 0..23 {
            if (seen >> i) & 1 == 0 {
                let mut j = i;
                let mut k = 0u8;
                let mut m = 0u32;
                while (seen >> j) & 1 == 0 {
                    seen |= 1 << j;
                    pos[j] = k;
                    m |= 1 << j;
                    k += 1;
                    j = x[j] as usize;
                }
                if k == 11 && nc < 2 {
                    cm[nc] = m;
                    nc += 1;
                }
            }
        }
        for &h in &self.heptads {
            for &c in &cm[..nc] {
                let inter = h & c;
                if inter.count_ones() >= 3 {
                    let mut s = 0u16;
                    let mut v = inter;
                    while v != 0 {
                        s |= 1 << pos[v.trailing_zeros() as usize];
                        v &= v - 1;
                    }
                    if self.a_only11[self.canon11[s as usize] as usize] {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// The transported system of a 23-cycle, compared with the two cyclic systems.
    fn steiner_sign_raw(&self, s: &Perm23) -> Result<Option<GolaySign>, M23Error> {
        if super::group::num_cycles(s) != 1 {
            return Err(M23Error::InvariantFailed(
                "cyclic_steiner_sign needs a 23-cycle".into(),
            ));
        }
        let mut lab = [0u8; 23];
        let mut x = 0usize;
        for i in 0..23 {
            lab[x] = i as u8;
            x = s[x] as usize;
        }
        let mut t: Vec<u32> = self.heptads.iter().map(|&h| mask_image(h, &lab)).collect();
        t.sort_unstable();
        Ok(if t == cyclic_golay::heptads(GolaySign::Plus) {
            Some(GolaySign::Plus)
        } else if t == cyclic_golay::heptads(GolaySign::Minus) {
            Some(GolaySign::Minus)
        } else {
            None
        })
    }

    /// Which cyclic Steiner system the 23-cycle `p` induces when the points are labelled
    /// `0^(p^i) -> i`. `Err(InvariantFailed)` if `p` is not a 23-cycle or the transported
    /// system is neither cyclic system (for example a 23-cycle outside M23).
    pub fn cyclic_steiner_sign(&self, p: &Perm23) -> Result<GolaySign, M23Error> {
        self.steiner_sign_raw(p)?.ok_or_else(|| {
            M23Error::InvariantFailed("the transported system is not a cyclic system".into())
        })
    }

    /// The class of `p`.
    ///
    /// Precondition: `p` lies in M23. On a non-member the result is unspecified: it may be
    /// `Ok` with a wrong class (an involution of type `2^8 1^7` outside M23 is called 2A, and
    /// a 22-cycle, which has the signature of the 11-cycles, is called 11B). Use
    /// [`ClassIdentifier::class_of_checked`] for input that may lie outside M23.
    /// `Err(NotInM23)` only for a signature outside the 12 types.
    #[inline]
    pub fn class_of_member(&self, p: &Perm23) -> Result<M23Class, M23Error> {
        let t = type_of(p).ok_or(M23Error::NotInM23)?;
        let cl = TYPE_CLASSES[t];
        let c = if cl.len() == 1 {
            cl[0]
        } else if t == T11 {
            if self.is_11a(p) {
                cl[0]
            } else {
                cl[1]
            }
        } else if self.trace_of_inverse(p) == self.tr_inv_a[t] {
            cl[0]
        } else {
            cl[1]
        };
        Ok(M23Class::ALL[c])
    }

    /// The class of `p` after a membership test by sifting: `Err(NotInM23)` for a non-member.
    pub fn class_of_checked(&self, g: &M23, p: &Perm23) -> Result<M23Class, M23Error> {
        if !g.contains(p) {
            return Err(M23Error::NotInM23);
        }
        self.class_of_member(p)
    }

    /// The representative of class `c`: `g2` for 23A, `h11` for 11A, the inverses for 23B and
    /// 11B, and an element found by a deterministic walk for the others.
    pub fn representative(&self, c: M23Class) -> Perm23 {
        self.reps[c.index()]
    }

    /// `|C_M23(rep(c))|`, computed by RustMath: sifting `C_Sym(23)(rep(c))` when it has at
    /// most [`SIFT_BUDGET`] elements (4A to 23B), otherwise `|G| / |class|` with the class as
    /// the conjugation orbit of `rep(c)`, capped at [`ORBIT_CAP`] (1A, 2A, 3A). Cached per
    /// class. Every class of M23 falls under one of the two, so an `Err` here is a defect.
    pub fn centralizer_order(&self, g: &M23, c: M23Class) -> Result<u64, M23Error> {
        if let Some(&v) = self.centralizers[c.index()].get() {
            return Ok(v);
        }
        let x = self.representative(c);
        let v = match centralizer_order_by_sifting(g, &x, SIFT_BUDGET) {
            Ok(k) => k,
            Err(M23Error::BudgetExceeded { .. }) => {
                let orbit = conjugacy_class_by_orbit(g, &x, ORBIT_CAP)?;
                let len = orbit.len() as u64;
                if !M23_ORDER.is_multiple_of(len) {
                    return Err(M23Error::Construction(format!(
                        "an orbit of length {len} does not divide |M23|"
                    )));
                }
                M23_ORDER / len
            }
            Err(e) => return Err(e),
        };
        let _ = self.centralizers[c.index()].set(v);
        Ok(v)
    }
}

/// `|C_M23(x)| = #{c in C_Sym(23)(x) : c in M23}`: enumerates `C_Sym(x)` (a product of wreath
/// products `C_l wr S_m` over the cycle lengths `l` with multiplicity `m`) and sifts each
/// element. Exact. `Err(BudgetExceeded)` when `|C_Sym(x)| = prod m! l^m` exceeds `budget`; the
/// size is computed with saturating arithmetic (`23!` for the identity does not fit in `u64`),
/// and a saturated size (`u64::MAX`) counts as above every budget, `u64::MAX` included.
/// `Err(NotInM23)` if `x` is not in M23.
pub fn centralizer_order_by_sifting(g: &M23, x: &Perm23, budget: u64) -> Result<u64, M23Error> {
    if !g.contains(x) {
        return Err(M23Error::NotInM23);
    }
    // cycles grouped by length, in increasing length
    let mut by_len: Vec<(usize, Vec<Vec<u8>>)> = Vec::new();
    for c in super::group::cycles(x) {
        match by_len.iter_mut().find(|(l, _)| *l == c.len()) {
            Some((_, v)) => v.push(c),
            None => by_len.push((c.len(), vec![c])),
        }
    }
    by_len.sort_by_key(|(l, _)| *l);
    let mut size: u64 = 1;
    for (l, cs) in &by_len {
        let m = cs.len() as u64;
        let fact = (1..=m).fold(1u64, |a, k| a.saturating_mul(k));
        size = size
            .saturating_mul(fact)
            .saturating_mul((*l as u64).saturating_pow(m as u32));
    }
    // u64::MAX is a saturated size: the true size may be larger than any budget
    if size == u64::MAX || size > budget {
        return Err(M23Error::BudgetExceeded {
            needed: size,
            budget,
        });
    }
    fn perms(m: usize) -> Vec<Vec<usize>> {
        if m == 0 {
            return vec![vec![]];
        }
        let mut out = Vec::new();
        for p in perms(m - 1) {
            for pos in 0..=p.len() {
                let mut q = p.clone();
                q.insert(pos, m - 1);
                out.push(q);
            }
        }
        out
    }
    // For each length: every permutation of its cycles with a rotation of each.
    let choices: Vec<Vec<(Vec<usize>, Vec<usize>)>> = by_len
        .iter()
        .map(|(l, cs)| {
            let m = cs.len();
            let mut v = Vec::new();
            for p in perms(m) {
                let mut rot = vec![0usize; m];
                loop {
                    v.push((p.clone(), rot.clone()));
                    let mut k = 0;
                    while k < m {
                        rot[k] += 1;
                        if rot[k] < *l {
                            break;
                        }
                        rot[k] = 0;
                        k += 1;
                    }
                    if k == m {
                        break;
                    }
                }
            }
            v
        })
        .collect();
    let mut idx = vec![0usize; by_len.len()];
    let mut count = 0u64;
    loop {
        let mut c = [0usize; 23];
        for (gi, (l, cs)) in by_len.iter().enumerate() {
            let (p, rot) = &choices[gi][idx[gi]];
            for (a, cyc) in cs.iter().enumerate() {
                let target = &cs[p[a]];
                for (s, &pt) in cyc.iter().enumerate() {
                    c[pt as usize] = target[(s + rot[a]) % l] as usize;
                }
            }
        }
        if g.chain().contains(&c) {
            count += 1;
        }
        let mut k = 0;
        loop {
            if k == by_len.len() {
                return Ok(count);
            }
            idx[k] += 1;
            if idx[k] < choices[k].len() {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// The conjugacy class of `x` in M23, as the orbit of `x` under conjugation by `g1` and `g2`
/// (breadth first; exact, since a finite orbit closed under the generators is closed under the
/// group). `Err(BudgetExceeded)` when the orbit has more than `cap` elements; `Err(NotInM23)` if
/// `x` is not in M23.
pub fn conjugacy_class_by_orbit(g: &M23, x: &Perm23, cap: usize) -> Result<Vec<Perm23>, M23Error> {
    if !g.contains(x) {
        return Err(M23Error::NotInM23);
    }
    let gens = [g.g1(), g.g2()];
    let mut seen = HashSet::from([*x]);
    let mut out = vec![*x];
    let mut i = 0;
    while i < out.len() {
        let y = out[i];
        i += 1;
        for h in &gens {
            let z = conjugate(&y, h);
            if seen.insert(z) {
                if out.len() == cap {
                    return Err(M23Error::BudgetExceeded {
                        needed: cap as u64 + 1,
                        budget: cap as u64,
                    });
                }
                out.push(z);
            }
        }
    }
    Ok(out)
}

/// The result of [`census`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Census {
    /// The number of elements in each class (ATLAS order), from the labels of all elements.
    pub class_sizes: [u64; 17],
    /// `|C(rep(c))|` for each class, by counting the elements that commute with the
    /// representative.
    pub centralizer_orders: [u64; 17],
    /// The number of elements of order 23 whose cyclic Steiner sign agrees with the class
    /// given by the Golay trace (23A has the sign of `g2`). All 887,040 in M23.
    pub sign_agrees_on_23: u64,
}

/// Labels all 10,200,960 elements: the 17 class sizes, the agreement of the two 23AB
/// invariants on every element of order 23 (same pass), and the 17 centralizer orders by a
/// second pass that counts the elements commuting with each representative. About 9 s in a
/// release build; much slower in debug.
pub fn census(g: &M23, id: &ClassIdentifier) -> Result<Census, M23Error> {
    let mut class_sizes = [0u64; 17];
    let sign_a = id.cyclic_steiner_sign(&g.g2())?;
    let mut agree = 0u64;
    let mut err: Option<M23Error> = None;
    g.for_each_element(|p| {
        if err.is_some() {
            return;
        }
        match id.class_of_member(p) {
            Ok(c) => {
                class_sizes[c.index()] += 1;
                if class_type(c) == T23 {
                    let want = if c == M23Class::C23A {
                        sign_a
                    } else {
                        sign_a.opposite()
                    };
                    match id.cyclic_steiner_sign(p) {
                        Ok(s) if s == want => agree += 1,
                        Ok(_) => {}
                        Err(e) => err = Some(e),
                    }
                }
            }
            Err(e) => err = Some(e),
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    let reps: Vec<Perm23> = M23Class::ALL
        .iter()
        .map(|&c| id.representative(c))
        .collect();
    let mut centralizer_orders = [0u64; 17];
    g.for_each_element(|p| {
        for (k, x) in reps.iter().enumerate() {
            if rmul(p, x) == rmul(x, p) {
                centralizer_orders[k] += 1;
            }
        }
    });
    Ok(Census {
        class_sizes,
        centralizer_orders,
        sign_agrees_on_23: agree,
    })
}

/// The elements of chosen classes, built only by enumeration or by conjugation orbits, so a
/// list holds exactly its class.
#[derive(Clone, Debug, Default)]
pub struct ClassLists {
    lists: Vec<Option<Vec<Perm23>>>,
}

impl ClassLists {
    /// One pass over all elements, keeping those of the chosen classes (about 0.8 s in release
    /// for the six rational classes; 73 MB for them). A stored element takes 23 bytes, so all
    /// 17 classes take at most 235 MB, plus the spare capacity of the growing vectors.
    pub fn by_enumeration(
        g: &M23,
        id: &ClassIdentifier,
        classes: &[M23Class],
    ) -> Result<ClassLists, M23Error> {
        let mut lists: Vec<Option<Vec<Perm23>>> = vec![None; 17];
        let mut want_type = [false; 12];
        for &c in classes {
            lists[c.index()] = Some(Vec::new());
            want_type[class_type(c)] = true;
        }
        let mut err = None;
        g.for_each_element(|p| {
            if err.is_some() {
                return;
            }
            match type_of(p) {
                Some(t) if want_type[t] => match id.class_of_member(p) {
                    Ok(c) => {
                        if let Some(v) = lists[c.index()].as_mut() {
                            v.push(*p);
                        }
                    }
                    Err(e) => err = Some(e),
                },
                Some(_) => {}
                None => err = Some(M23Error::NotInM23),
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(ClassLists { lists }),
        }
    }

    /// The chosen classes as conjugation orbits of their representatives, each capped at `cap`
    /// elements (`Err(BudgetExceeded)` above). For small classes (2A: 3,795, 3A: 56,672) in
    /// debug builds.
    pub fn by_orbit(
        g: &M23,
        id: &ClassIdentifier,
        classes: &[M23Class],
        cap: usize,
    ) -> Result<ClassLists, M23Error> {
        let mut lists: Vec<Option<Vec<Perm23>>> = vec![None; 17];
        for &c in classes {
            if lists[c.index()].is_none() {
                lists[c.index()] = Some(conjugacy_class_by_orbit(g, &id.representative(c), cap)?);
            }
        }
        Ok(ClassLists { lists })
    }

    /// The elements of class `c`, if that class was built.
    pub fn get(&self, c: M23Class) -> Option<&[Perm23]> {
        self.lists.get(c.index())?.as_deref()
    }
}
