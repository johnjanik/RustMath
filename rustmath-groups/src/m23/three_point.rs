//! The three-point class vectors of M23 that can carry a cover over Q: enumeration, the exact
//! count of Nielsen classes, generation by transitivity, real points, genus (WP-P, HANDOFF
//! 4.1 and 4.2 steps 4 to 6).
//!
//! MAGMA: ch. 58 (permutation groups, `IsTransitive`), ch. 91 (structure constants); digests
//! `rustmath-lava/docs/Chapter_58_Permutation_Groups.md`,
//! `Chapter_91_Characters_of_Finite_Groups.md`.
//!
//! Sources:
//! - The owner's `scripts/passports_all.py` and `scripts/gap/m23_passports.g`
//!   (`johnjanik/M23` at `2d11038`): the enumeration and the counting, ported as algorithms.
//! - Serre, *Topics in Galois Theory* (1992), 7.2: `n = #{(g1, g2, g3) in C1 x C2 x C3 :
//!   g1 g2 g3 = 1} / |G|`.
//! - ATLAS of Finite Groups (1985), M23, p. 71: the maximal subgroups M22, L3(4):2, 2^4:A7, A8,
//!   M11, 2^4:(3 x A5):2, 23:11; all but 23:11 are intransitive on the 23 points (checked with
//!   GAP 4.12.1, `MaximalSubgroupClassReps(MathieuGroup(23))`, design WPP.md 5.2).
//! - HJLPPZ (arXiv:2608.08538v2), 3.1, and the owner's paper, Section 2: complex conjugation on
//!   `(R, D, D^-1)` at `0, s, conj(s)` acts by `(g1, g2, g3) -> (g1^-1, g3^-1, g2^-1)`.
//!
//! # The counting (HANDOFF 4.2 step 4)
//!
//! Fix `b` in the class `C_b`. The survivors are the `a` in `C_a` with `a b` in the third
//! class (for `(R, D, D^-1)`: `b = rep(D)`, `a` in `R`, `a b` in `D`). The triples
//! `(a, b, (ab)^-1)` with `b` fixed number `n |C(b)|`, so `#survivors = n |C(b)|` exactly, with
//! `n` from the table and `|C(b)|` computed by RustMath (`ClassIdentifier::centralizer_order`).
//! A failure is `Err(CountIdentityFailed)`. Since `C(b)` acts freely on the generating pairs,
//! `size = #(generating survivors) / |C(b)|`.
//!
//! # Generation (step 5)
//!
//! By the ATLAS list, a transitive proper subgroup of M23 lies in a conjugate of 23:11, whose
//! elements have order 1, 11 or 23. So a subgroup that contains an element of order outside
//! `{1, 11, 23}` is M23 if and only if it is transitive. Every class vector with a class of such
//! an order is decided by the transitivity of `<a, b>`; for any other vector the count refuses
//! with `Err(GenerationCriterionNotApplicable)` (a bounded search would not decide).
//!
//! # Real points (step 6)
//!
//! Conjugate pairs `(R, D, D^-1)`: a class with `g2 = b` is fixed by complex conjugation iff
//! some `c` in M23 has `c a c^-1 = a^-1` and `c b c^-1 = a b`; then `c^2` centralizes `<a, b>`,
//! so `c` is an involution, unique, and `a = c b c b^-1`. Real classes =
//! `#{c in 2A : c b c b^-1 in R, <c b c b^-1, b> = M23} / |C(b)|`.
//!
//! Rational vectors, three real branch points. The classical statements are in Fried, Comm.
//! Algebra 5 (1977), Krull and Neukirch, Math. Ann. 193 (1971), and Debes and Fried, Pacific J.
//! Math. 163 (1994); they were not read for this port, so the derivation is written here and
//! every count records [`TrustedResult::ConjugationThreeRealPoints`]. Let the branch points be
//! `t1 < t2 < t3` in `P^1(R)` (after a real Moebius map all are finite) and take the base point
//! `p` in `(t1, t2)`. Let `gamma_1` run from `p` along the real segment to near `t1`, once
//! around `t1` on a small circle, and back; `gamma_2` the same around `t2`. They generate
//! `pi_1(P^1 - {t1, t2, t3}, p)`. Complex conjugation `kappa` fixes `p` and both segments and
//! reflects each small circle, so `kappa(gamma_i) = gamma_i^-1`. A cover with monodromy `phi`
//! has conjugate with monodromy `phi o kappa_*` (the base point is real), so conjugation sends
//! the class of `(a, b)` to that of `(a^-1, b^-1)`. The Nielsen classes up to conjugation in M23
//! are the absolute ones, since `N_{S_23}(M23) = M23`. The class is fixed iff some `h` inverts
//! `a` and `b`; then `h^2` centralizes `<a, b> = M23`, so `h^2 = 1`, and `h != 1` (two
//! involutions generate a dihedral group), so `h = c` is an involution, and it is unique. Write
//! `a = c u`: then `u^2 = 1`. Real classes = `#{(c, u) : c in 2A, c b c = b^-1, u in {1} u 2A,
//! c u in C_a, c u b in C_c, <c u, b> = M23} / |C(b)|`. This is the owner's criterion
//! (`passports_all.py`, step 6), except that `u = 1` (`a = c`) is also enumerated; it never
//! adds, since `<c, b>` is then dihedral of order at most 16 and not transitive. The number of
//! fixed classes does not depend on which two branch points bound the arc of the base point, so
//! any choice of the roles of `a` and `b` gives the same count.

use std::fmt;

use rustmath_rationals::Rational;

use super::atlas::AtlasTable;
use super::classes::{class_type, type_of, ClassIdentifier, ClassLists, M23Class};
use super::group::{generates_transitive, inverse, num_cycles, rmul, M23Error, Perm23, M23};

/// The two shapes of class vector that can carry a cover over Q (branch cycle lemma,
/// HANDOFF 4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VectorKind {
    /// Three rational classes at three rational points.
    Rational,
    /// `(R, D, D^-1)`: a rational class at a rational point and an irrational pair at two
    /// conjugate points.
    ConjugatePair,
}

impl VectorKind {
    /// The name used in the owner's files: `"rational"` or `"conjugate_pair"`.
    pub fn name(self) -> &'static str {
        match self {
            VectorKind::Rational => "rational",
            VectorKind::ConjugatePair => "conjugate_pair",
        }
    }
}

/// A class vector `(C1, C2, C3)` of the given kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassVector {
    /// Rational or conjugate pair.
    pub kind: VectorKind,
    /// The three classes; for a conjugate pair `(R, D, D^-1)`.
    pub classes: [M23Class; 3],
}

impl ClassVector {
    /// Checks the shape (review 2, n1): `Rational` needs three rational classes;
    /// `ConjugatePair` needs `classes[0]` rational, `classes[1]` not rational and
    /// `classes[2] = classes[1]^-1`. Anything else is `Err(InvalidClassVector)`.
    pub fn validate(&self) -> Result<(), ThreePointError> {
        let [c0, c1, c2] = self.classes;
        let ok = match self.kind {
            VectorKind::Rational => c0.is_rational() && c1.is_rational() && c2.is_rational(),
            VectorKind::ConjugatePair => {
                c0.is_rational() && !c1.is_rational() && c2 == c1.inverse()
            }
        };
        if ok {
            Ok(())
        } else {
            Err(ThreePointError::InvalidClassVector(format!(
                "{} is not a valid {} vector",
                self,
                self.kind.name()
            )))
        }
    }
}

impl fmt::Display for ClassVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, {}, {})",
            self.classes[0], self.classes[1], self.classes[2]
        )
    }
}

/// Errors of the three-point counting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThreePointError {
    /// The class vector has the wrong shape for its kind (review 2, n1).
    InvalidClassVector(String),
    /// A class list needed by the count was not built.
    ClassListMissing(M23Class),
    /// `#survivors != n |C(b)|`: the class identification, the centralizer order or the table
    /// is wrong. Never a silent mismatch.
    CountIdentityFailed {
        /// The number of survivors counted.
        survivors: u64,
        /// The structure constant from the table.
        n: Rational,
        /// `|C(b)|` as computed.
        centralizer: u64,
    },
    /// A count that must be divisible (by `|C(b)|`, or `size - real` by 2) is not.
    NotDivisible(&'static str),
    /// All three classes have order in `{1, 11, 23}`: transitivity does not decide generation.
    GenerationCriterionNotApplicable,
    /// The table gave an error (a structure constant that is not rational).
    Table(String),
    /// An error of the group layer.
    Group(M23Error),
}

impl fmt::Display for ThreePointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThreePointError::InvalidClassVector(s) => write!(f, "{s}"),
            ThreePointError::ClassListMissing(c) => {
                write!(f, "the list of class {c} was not built")
            }
            ThreePointError::CountIdentityFailed {
                survivors,
                n,
                centralizer,
            } => write!(
                f,
                "#survivors = {survivors} but n |C(b)| = {n} * {centralizer}"
            ),
            ThreePointError::NotDivisible(s) => write!(f, "not divisible: {s}"),
            ThreePointError::GenerationCriterionNotApplicable => write!(
                f,
                "every class has order 1, 11 or 23: transitivity does not decide generation"
            ),
            ThreePointError::Table(s) => write!(f, "character table: {s}"),
            ThreePointError::Group(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ThreePointError {}

impl From<M23Error> for ThreePointError {
    fn from(e: M23Error) -> ThreePointError {
        ThreePointError::Group(e)
    }
}

/// The class vectors of nonnegative genus with `n > 0`, as the owner's GAP script
/// `m23_passports.g`: rational vectors `R1 <= R2 <= R3` over the rational classes except 1A
/// (the classes with rational values on every irreducible), then `(R, D, D^-1)` for each
/// rational `R` and each irrational pair with `D` the A class. For M23 these are the 57 class
/// vectors of HANDOFF 4.1.
pub fn enumerate_class_vectors(
    t: &AtlasTable,
    id: &ClassIdentifier,
) -> Result<Vec<ClassVector>, ThreePointError> {
    let rat: Vec<M23Class> = M23Class::ALL
        .iter()
        .copied()
        .filter(|&c| c != M23Class::C1A && t.is_rational_class(c))
        .collect();
    let pairs: Vec<M23Class> = M23Class::ALL
        .iter()
        .copied()
        .filter(|&c| !t.is_rational_class(c))
        .filter(|&c| t.power_map(-1, c).is_some_and(|d| c.index() < d.index()))
        .collect();
    let mut out = Vec::new();
    let mut keep = |v: ClassVector| -> Result<(), ThreePointError> {
        let n = t
            .structure_constant(v.classes)
            .map_err(ThreePointError::Table)?;
        if n > Rational::from(0i64) && genus23(id, &v)? >= 0 {
            out.push(v);
        }
        Ok(())
    };
    for (i, &a) in rat.iter().enumerate() {
        for (j, &b) in rat.iter().enumerate().skip(i) {
            for &c in &rat[j..] {
                keep(ClassVector {
                    kind: VectorKind::Rational,
                    classes: [a, b, c],
                })?;
            }
        }
    }
    for &r in &rat {
        for &d in &pairs {
            keep(ClassVector {
                kind: VectorKind::ConjugatePair,
                classes: [r, d, d.inverse()],
            })?;
        }
    }
    Ok(out)
}

/// The genus of the degree 23 cover, by Riemann-Hurwitz:
/// `2g - 2 = -46 + sum_i (23 - #cycles(rep(C_i)))`, with the cycles of RustMath's
/// representatives. `Err(NotDivisible)` if the right side is odd (impossible for a vector of
/// even permutations).
pub fn genus23(id: &ClassIdentifier, v: &ClassVector) -> Result<i64, ThreePointError> {
    let ram: i64 = v
        .classes
        .iter()
        .map(|&c| 23 - num_cycles(&id.representative(c)) as i64)
        .sum();
    let two_g = ram - 46 + 2;
    if two_g % 2 != 0 {
        return Err(ThreePointError::NotDivisible(
            "the Riemann-Hurwitz sum is odd",
        ));
    }
    Ok(two_g / 2)
}

/// How a count is decided. `Decided` is the only success status: every number is an exact
/// count and generation is decided by the theorem. When the theorem does not apply the call is
/// an `Err`, never a size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CountStatus {
    /// Exact count; generation by transitivity (ATLAS maximal subgroups).
    Decided,
}

/// The results a count rests on besides exact enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustedResult {
    /// ATLAS p. 71: every maximal subgroup of M23 except 23:11 is intransitive.
    AtlasMaximalSubgroups,
    /// HJLPPZ 3.1: conjugation on `(R, D, D^-1)` at `0, s, conj(s)` is
    /// `(g1, g2, g3) -> (g1^-1, g3^-1, g2^-1)`.
    ConjugationConjugatePair,
    /// Three real branch points: conjugation sends `(a, b)` to `(a^-1, b^-1)`. The derivation
    /// is in the header of `three_point.rs`; the classical sources are not quoted (WPP-7).
    ConjugationThreeRealPoints,
}

impl TrustedResult {
    /// A short label for printing.
    pub fn label(self) -> &'static str {
        match self {
            TrustedResult::AtlasMaximalSubgroups => "ATLAS maximal subgroups",
            TrustedResult::ConjugationConjugatePair => "conjugation (R,D,D') HJLPPZ 3.1",
            TrustedResult::ConjugationThreeRealPoints => "conjugation, three real points (derived)",
        }
    }
}

/// The survivors of a class vector, without deciding generation (review 2, n2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructureCount {
    /// The class vector.
    pub vector: ClassVector,
    /// `n` from the table.
    pub structure_constant: Rational,
    /// The class of the fixed `b`.
    pub b_class: M23Class,
    /// The class over which `a` runs.
    pub a_class: M23Class,
    /// `|C(b)|`, computed by RustMath (`ClassIdentifier::centralizer_order`).
    pub centralizer_order: u64,
    /// The number of `a` with `a b` in the third class; equal to `n |C(b)|` (checked).
    pub survivors: u64,
}

/// One class vector, counted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassVectorCount {
    /// The class vector.
    pub vector: ClassVector,
    /// The genus of the degree 23 cover.
    pub genus23: i64,
    /// `n` from the table.
    pub structure_constant: Rational,
    /// The class of the fixed `b`.
    pub b_class: M23Class,
    /// The class over which `a` runs.
    pub a_class: M23Class,
    /// `|C(b)|` from `ClassIdentifier::centralizer_order` (sifting or orbit), not from the
    /// table.
    pub centralizer_order: u64,
    /// The number of `a` with `a b` in the third class (`= n |C(b)|`, checked).
    pub survivors: u64,
    /// The survivors with `<a, b>` transitive, hence `= M23`.
    pub generating: u64,
    /// The number of Nielsen classes of generating triples, `generating / |C(b)|`.
    pub size: u64,
    /// Rational vectors: the generating pairs `(c, u)` with `u` in `{1} u 2A`; conjugate pairs:
    /// the involutions `c` with `<c b c b^-1, b> = M23`.
    pub real_involutions: u64,
    /// The number of classes fixed by complex conjugation, `real_involutions / |C(b)|`.
    pub real: u64,
    /// Rational vectors with the `a` class 2A: the `c` with `a = c` (`u = 1`) whose product
    /// with `b` lies in the third class.
    pub a_equals_c_candidates: u64,
    /// Those of them that generate (0 by the dihedral argument; asserted by the tests).
    pub a_equals_c_generating: u64,
    /// Always `Decided` on `Ok`.
    pub status: CountStatus,
    /// The results the count rests on besides exact enumeration.
    pub trusted: Vec<TrustedResult>,
}

/// The roles: `(a_class, b_class, c_class)` with survivors `a` in `C_a`, `a b` in `C_c`.
fn roles(t: &AtlasTable, v: &ClassVector) -> (M23Class, M23Class, M23Class) {
    match v.kind {
        VectorKind::ConjugatePair => (v.classes[0], v.classes[1], v.classes[1]),
        VectorKind::Rational => {
            // b in the class with the smallest centralizer, a in the smaller of the other two
            // (any choice gives a correct count; this one is the fastest).
            let mut cl = v.classes;
            cl.sort_by_key(|&c| (t.centralizer_order(c), c));
            let (x, y) = (cl[1], cl[2]);
            let (a, c) = if t.class_size(x) <= t.class_size(y) {
                (x, y)
            } else {
                (y, x)
            };
            (a, cl[0], c)
        }
    }
}

/// Whether `p` lies in class `c` (exact for members of M23).
#[inline]
fn in_class(id: &ClassIdentifier, p: &Perm23, c: M23Class) -> bool {
    match type_of(p) {
        Some(t) if t == class_type(c) => {
            c.is_rational() || id.class_of_member(p).map(|x| x == c).unwrap_or(false)
        }
        _ => false,
    }
}

/// The survivors and the generating survivors.
fn survey(
    g: &M23,
    id: &ClassIdentifier,
    t: &AtlasTable,
    lists: &ClassLists,
    v: &ClassVector,
) -> Result<(StructureCount, u64), ThreePointError> {
    v.validate()?;
    let n = t
        .structure_constant(v.classes)
        .map_err(ThreePointError::Table)?;
    let (a_class, b_class, c_class) = roles(t, v);
    let a_list = lists
        .get(a_class)
        .ok_or(ThreePointError::ClassListMissing(a_class))?;
    let centralizer = id.centralizer_order(g, b_class)?;
    let b = id.representative(b_class);
    let (mut survivors, mut generating) = (0u64, 0u64);
    for a in a_list {
        let p = rmul(a, &b);
        if in_class(id, &p, c_class) {
            survivors += 1;
            if generates_transitive(a, &b) {
                generating += 1;
            }
        }
    }
    if Rational::from(survivors as i64) != &n * &Rational::from(centralizer as i64) {
        return Err(ThreePointError::CountIdentityFailed {
            survivors,
            n,
            centralizer,
        });
    }
    Ok((
        StructureCount {
            vector: *v,
            structure_constant: n,
            b_class,
            a_class,
            centralizer_order: centralizer,
            survivors,
        },
        generating,
    ))
}

/// The survivors of `v` with the identity `#survivors = n |C(b)|` checked, without deciding
/// generation, so it works for every valid class vector. `lists` must hold the `a` class.
pub fn structure_count(
    g: &M23,
    id: &ClassIdentifier,
    t: &AtlasTable,
    lists: &ClassLists,
    v: &ClassVector,
) -> Result<StructureCount, ThreePointError> {
    survey(g, id, t, lists, v).map(|(s, _)| s)
}

/// Counts one class vector: survivors (with the identity), generating survivors (by
/// transitivity), the size, the real classes, the genus. `|C(b)|` comes from
/// `id.centralizer_order(g, b_class)` (cached). `lists` must hold the `a` class and 2A.
pub fn count_class_vector(
    g: &M23,
    id: &ClassIdentifier,
    t: &AtlasTable,
    lists: &ClassLists,
    v: &ClassVector,
) -> Result<ClassVectorCount, ThreePointError> {
    let (sc, generating) = survey(g, id, t, lists, v)?;
    if v.classes
        .iter()
        .all(|c| [1, 11, 23].contains(&c.element_order()))
    {
        return Err(ThreePointError::GenerationCriterionNotApplicable);
    }
    let genus = genus23(id, v)?;
    let cent = sc.centralizer_order;
    if generating % cent != 0 {
        return Err(ThreePointError::NotDivisible("generating pairs by |C(b)|"));
    }
    let inv2 = lists
        .get(M23Class::C2A)
        .ok_or(ThreePointError::ClassListMissing(M23Class::C2A))?;
    let b = id.representative(sc.b_class);
    let binv = inverse(&b);
    let (mut real_involutions, mut ac_cand, mut ac_gen) = (0u64, 0u64, 0u64);
    let mut trusted = vec![TrustedResult::AtlasMaximalSubgroups];
    match v.kind {
        VectorKind::ConjugatePair => {
            trusted.push(TrustedResult::ConjugationConjugatePair);
            let r = v.classes[0];
            for c in inv2 {
                let a = rmul(&rmul(&rmul(c, &b), c), &binv);
                if in_class(id, &a, r) && generates_transitive(&a, &b) {
                    real_involutions += 1;
                }
            }
        }
        VectorKind::Rational => {
            trusted.push(TrustedResult::ConjugationThreeRealPoints);
            let (a_class, c_class) = (sc.a_class, roles(t, v).2);
            for c in inv2 {
                if rmul(&rmul(c, &b), c) != binv {
                    continue;
                }
                if a_class == M23Class::C2A {
                    // u = 1, a = c
                    if in_class(id, &rmul(c, &b), c_class) {
                        ac_cand += 1;
                        if generates_transitive(c, &b) {
                            ac_gen += 1;
                            real_involutions += 1;
                        }
                    }
                }
                for u in inv2 {
                    let a = rmul(c, u);
                    if !in_class(id, &a, a_class) {
                        continue;
                    }
                    if in_class(id, &rmul(&a, &b), c_class) && generates_transitive(&a, &b) {
                        real_involutions += 1;
                    }
                }
            }
        }
    }
    if real_involutions % cent != 0 {
        return Err(ThreePointError::NotDivisible("real pairs by |C(b)|"));
    }
    let (size, real) = (generating / cent, real_involutions / cent);
    if real > size || (size - real) % 2 != 0 {
        return Err(ThreePointError::NotDivisible("size - real by 2"));
    }
    Ok(ClassVectorCount {
        vector: *v,
        genus23: genus,
        structure_constant: sc.structure_constant,
        b_class: sc.b_class,
        a_class: sc.a_class,
        centralizer_order: cent,
        survivors: sc.survivors,
        generating,
        size,
        real_involutions,
        real,
        a_equals_c_candidates: ac_cand,
        a_equals_c_generating: ac_gen,
        status: CountStatus::Decided,
        trusted,
    })
}
