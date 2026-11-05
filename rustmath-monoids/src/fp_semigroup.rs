//! Finitely presented semigroups and monoids (MAGMA handbook chapter 77,
//! §77.4–77.7).
//!
//! An fp-semigroup is a quotient of a free semigroup by a set of relations
//! `w1 = w2`. In this chapter the objects are *syntactic*: relations are stored
//! but words are compared as elements of the underlying free semigroup, not
//! modulo the relations (semantic reduction modulo relations is the province of
//! rewrite monoids, chapter 78 — see [`crate::rws_monoid`]).
//!
//! This module provides presentation specification (`Semigroup<…>`,
//! `Monoid<…>`, `quo<…>`), relation/generator access, the elementary Tietze
//! transformations (§77.7), sub/ideal descriptors (§77.5) and the extension
//! constructors `DirectProduct` / `FreeProduct` (§77.6).
//!
//! Generator indices are 0-based.

use crate::free_semigroup::FreeSemigroupElement;

/// A relation `lhs = rhs` between words over the generators (§77.4.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    lhs: Vec<usize>,
    rhs: Vec<usize>,
}

impl Relation {
    /// Create the relation `w1 = w2`.
    pub fn new(lhs: Vec<usize>, rhs: Vec<usize>) -> Self {
        Relation { lhs, rhs }
    }

    /// Create a relation from two free-semigroup words.
    pub fn from_words(lhs: &FreeSemigroupElement, rhs: &FreeSemigroupElement) -> Self {
        Relation {
            lhs: lhs.word().to_vec(),
            rhs: rhs.word().to_vec(),
        }
    }

    /// `LHS(r)`: the left-hand side.
    pub fn lhs(&self) -> &[usize] {
        &self.lhs
    }

    /// `RHS(r)`: the right-hand side.
    pub fn rhs(&self) -> &[usize] {
        &self.rhs
    }

    fn mentions(&self, g: usize) -> bool {
        self.lhs.contains(&g) || self.rhs.contains(&g)
    }

    fn reindex(&self, removed: usize) -> Relation {
        let map = |w: &[usize]| {
            w.iter()
                .map(|&x| if x > removed { x - 1 } else { x })
                .collect()
        };
        Relation {
            lhs: map(&self.lhs),
            rhs: map(&self.rhs),
        }
    }
}

/// A finitely presented semigroup or monoid.
#[derive(Debug, Clone)]
pub struct FpSemigroup {
    num_generators: usize,
    generator_names: Vec<String>,
    relations: Vec<Relation>,
    is_monoid: bool,
}

impl FpSemigroup {
    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("x_{}", i + 1)).collect()
    }

    /// A free semigroup presentation on `n` generators (no relations).
    pub fn free_semigroup(n: usize) -> Self {
        FpSemigroup {
            num_generators: n,
            generator_names: Self::names(n),
            relations: Vec::new(),
            is_monoid: false,
        }
    }

    /// A free monoid presentation on `n` generators (no relations).
    pub fn free_monoid(n: usize) -> Self {
        FpSemigroup {
            num_generators: n,
            generator_names: Self::names(n),
            relations: Vec::new(),
            is_monoid: true,
        }
    }

    /// `Semigroup< x1, …, xr | relations >`: the quotient of the free semigroup
    /// on `n` generators by `relations`.
    pub fn semigroup(n: usize, relations: Vec<Relation>) -> Self {
        FpSemigroup {
            num_generators: n,
            generator_names: Self::names(n),
            relations,
            is_monoid: false,
        }
    }

    /// `Monoid< x1, …, xr | relations >`: the quotient of the free monoid on
    /// `n` generators by `relations`.
    pub fn monoid(n: usize, relations: Vec<Relation>) -> Self {
        FpSemigroup {
            num_generators: n,
            generator_names: Self::names(n),
            relations,
            is_monoid: true,
        }
    }

    /// `quo< F | relations >`: form the quotient of `self` by adding
    /// `relations` (§77.5.2).
    pub fn quo(&self, mut relations: Vec<Relation>) -> Self {
        let mut all = self.relations.clone();
        all.append(&mut relations);
        FpSemigroup {
            num_generators: self.num_generators,
            generator_names: self.generator_names.clone(),
            relations: all,
            is_monoid: self.is_monoid,
        }
    }

    /// Whether this presentation is a monoid (has an identity).
    pub fn is_monoid(&self) -> bool {
        self.is_monoid
    }

    /// `NumberOfGenerators` / `Ngens`.
    pub fn ngens(&self) -> usize {
        self.num_generators
    }

    /// The generator names.
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// The `i`-th generator `S.i` as a word.
    pub fn gen(&self, i: usize) -> Option<FreeSemigroupElement> {
        if i < self.num_generators {
            Some(FreeSemigroupElement::new(vec![i]))
        } else {
            None
        }
    }

    /// `Generators(S)`.
    pub fn generators(&self) -> Vec<FreeSemigroupElement> {
        (0..self.num_generators)
            .map(|i| FreeSemigroupElement::new(vec![i]))
            .collect()
    }

    /// `Relations(S)`.
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    /// `NumberOfRelations` / `Nrels`.
    pub fn nrels(&self) -> usize {
        self.relations.len()
    }

    // ----- 77.7 Elementary Tietze transformations (each returns a new object) -----

    /// `AddRelation(S, r)`: append `r` to the defining relations.
    pub fn add_relation(&self, r: Relation) -> Self {
        let mut t = self.clone();
        t.relations.push(r);
        t
    }

    /// `AddRelation(S, r, i)`: insert `r` after the `i`-th relation (1-based).
    pub fn add_relation_at(&self, r: Relation, i: usize) -> Self {
        let mut t = self.clone();
        let pos = i.min(t.relations.len());
        t.relations.insert(pos, r);
        t
    }

    /// `DeleteRelation(S, r)`: remove the (first) matching relation.
    pub fn delete_relation(&self, r: &Relation) -> Self {
        let mut t = self.clone();
        if let Some(pos) = t.relations.iter().position(|x| x == r) {
            t.relations.remove(pos);
        }
        t
    }

    /// `DeleteRelation(S, i)`: remove the `i`-th relation (1-based).
    pub fn delete_relation_at(&self, i: usize) -> Self {
        let mut t = self.clone();
        if i >= 1 && i <= t.relations.len() {
            t.relations.remove(i - 1);
        }
        t
    }

    /// `ReplaceRelation(S, r1, r2)`: replace defining relation `r1` by `r2`.
    pub fn replace_relation(&self, r1: &Relation, r2: Relation) -> Self {
        let mut t = self.clone();
        if let Some(pos) = t.relations.iter().position(|x| x == r1) {
            t.relations[pos] = r2;
        }
        t
    }

    /// `ReplaceRelation(S, i, r)`: replace the `i`-th relation (1-based) by `r`.
    pub fn replace_relation_at(&self, i: usize, r: Relation) -> Self {
        let mut t = self.clone();
        if i >= 1 && i <= t.relations.len() {
            t.relations[i - 1] = r;
        }
        t
    }

    /// `AddGenerator(S)`: add a fresh generator `y` (presentation `⟨X∪{y}|R⟩`).
    pub fn add_generator(&self) -> Self {
        let mut t = self.clone();
        t.generator_names.push(format!("x_{}", t.num_generators + 1));
        t.num_generators += 1;
        t
    }

    /// `AddGenerator(S, w)`: add a fresh generator `y` with the relation
    /// `y = w` (presentation `⟨X∪{y}|R∪{y=w}⟩`).
    pub fn add_generator_with(&self, w: &FreeSemigroupElement) -> Self {
        let mut t = self.add_generator();
        let y = t.num_generators - 1;
        t.relations.push(Relation::new(vec![y], w.word().to_vec()));
        t
    }

    /// `DeleteGenerator(S, y)`: remove generator `y`, provided it occurs in at
    /// most one relation (which is also removed), reindexing the rest.
    pub fn delete_generator(&self, y: usize) -> Self {
        let mut relations: Vec<Relation> = self
            .relations
            .iter()
            .filter(|r| !r.mentions(y))
            .map(|r| r.reindex(y))
            .collect();
        // relations that mentioned y are dropped (per the precondition there is
        // at most one such relation)
        relations.dedup();
        let mut generator_names = self.generator_names.clone();
        if y < generator_names.len() {
            generator_names.remove(y);
        }
        FpSemigroup {
            num_generators: self.num_generators.saturating_sub(1),
            generator_names,
            relations,
            is_monoid: self.is_monoid,
        }
    }

    // ----- 77.6 Extensions -----

    /// `DirectProduct(R, S)`: generators of both, relations of both, plus
    /// commuting relations `r * s = s * r` for every `r ∈ R`, `s ∈ S`.
    pub fn direct_product(&self, other: &FpSemigroup) -> Self {
        let shift = self.num_generators;
        let mut relations = self.relations.clone();
        for r in &other.relations {
            relations.push(shift_relation(r, shift));
        }
        for i in 0..self.num_generators {
            for j in 0..other.num_generators {
                let s = shift + j;
                relations.push(Relation::new(vec![i, s], vec![s, i]));
            }
        }
        let mut generator_names = self.generator_names.clone();
        generator_names.extend(other.generator_names.iter().cloned());
        FpSemigroup {
            num_generators: self.num_generators + other.num_generators,
            generator_names,
            relations,
            is_monoid: self.is_monoid && other.is_monoid,
        }
    }

    /// `FreeProduct(R, S)`: disjoint union of generators and relations (no
    /// interaction between the two factors).
    pub fn free_product(&self, other: &FpSemigroup) -> Self {
        let shift = self.num_generators;
        let mut relations = self.relations.clone();
        for r in &other.relations {
            relations.push(shift_relation(r, shift));
        }
        let mut generator_names = self.generator_names.clone();
        generator_names.extend(other.generator_names.iter().cloned());
        FpSemigroup {
            num_generators: self.num_generators + other.num_generators,
            generator_names,
            relations,
            is_monoid: self.is_monoid && other.is_monoid,
        }
    }
}

fn shift_relation(r: &Relation, shift: usize) -> Relation {
    let map = |w: &[usize]| w.iter().map(|&x| x + shift).collect();
    Relation {
        lhs: map(&r.lhs),
        rhs: map(&r.rhs),
    }
}

/// The kind of a sub-structure descriptor (§77.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubKind {
    /// `sub< S | … >`
    Subsemigroup,
    /// `ideal< S | … >`
    TwoSidedIdeal,
    /// `lideal< S | … >`
    LeftIdeal,
    /// `rideal< S | … >`
    RightIdeal,
}

/// A sub-structure (subsemigroup or ideal) of an fp-semigroup, recorded by its
/// generating words (§77.5.1).
#[derive(Debug, Clone)]
pub struct SubStructure {
    kind: SubKind,
    generators: Vec<Vec<usize>>,
}

impl SubStructure {
    /// The kind of sub-structure.
    pub fn kind(&self) -> SubKind {
        self.kind
    }

    /// The generating words.
    pub fn generators(&self) -> &[Vec<usize>] {
        &self.generators
    }
}

fn collect_words(words: &[FreeSemigroupElement]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    for w in words {
        if w.is_empty() {
            continue; // identity is removed unless the whole thing is trivial
        }
        let v = w.word().to_vec();
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

impl FpSemigroup {
    /// `sub< S | … >`: the subsemigroup generated by the given words.
    pub fn sub(&self, generators: &[FreeSemigroupElement]) -> SubStructure {
        SubStructure {
            kind: SubKind::Subsemigroup,
            generators: collect_words(generators),
        }
    }

    /// `ideal< S | … >`: the two-sided ideal generated by the given words.
    pub fn ideal(&self, generators: &[FreeSemigroupElement]) -> SubStructure {
        SubStructure {
            kind: SubKind::TwoSidedIdeal,
            generators: collect_words(generators),
        }
    }

    /// `lideal< S | … >`: the left ideal generated by the given words.
    pub fn lideal(&self, generators: &[FreeSemigroupElement]) -> SubStructure {
        SubStructure {
            kind: SubKind::LeftIdeal,
            generators: collect_words(generators),
        }
    }

    /// `rideal< S | … >`: the right ideal generated by the given words.
    pub fn rideal(&self, generators: &[FreeSemigroupElement]) -> SubStructure {
        SubStructure {
            kind: SubKind::RightIdeal,
            generators: collect_words(generators),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presentation_from_h77e2() {
        // MAGMA H77E2: Monoid< x, y | x^2, y^2, (xy)^2 >  (relators = identity).
        let m = FpSemigroup::monoid(
            2,
            vec![
                Relation::new(vec![0, 0], vec![]),
                Relation::new(vec![1, 1], vec![]),
                Relation::new(vec![0, 1, 0, 1], vec![]),
            ],
        );
        assert_eq!(m.ngens(), 2);
        assert_eq!(m.nrels(), 3);
        assert!(m.is_monoid());
        assert_eq!(m.relations()[0].lhs(), &[0, 0]);
        assert_eq!(m.relations()[2].lhs(), &[0, 1, 0, 1]);
    }

    #[test]
    fn tietze_add_delete_replace_relation() {
        let s = FpSemigroup::semigroup(2, vec![Relation::new(vec![0, 0], vec![0])]);
        let r2 = Relation::new(vec![1, 1], vec![1]);
        let s2 = s.add_relation(r2.clone());
        assert_eq!(s2.nrels(), 2);
        // delete by value
        let s3 = s2.delete_relation(&r2);
        assert_eq!(s3.nrels(), 1);
        // replace by index
        let s4 = s2.replace_relation_at(1, Relation::new(vec![0], vec![0, 0]));
        assert_eq!(s4.relations()[0].lhs(), &[0]);
        // delete by index
        let s5 = s2.delete_relation_at(1);
        assert_eq!(s5.relations()[0].lhs(), &[1, 1]);
    }

    #[test]
    fn tietze_add_delete_generator() {
        let s = FpSemigroup::semigroup(2, vec![Relation::new(vec![0, 1], vec![1, 0])]);
        // add generator y = x1*x2
        let t = s.add_generator_with(&FreeSemigroupElement::new(vec![0, 1]));
        assert_eq!(t.ngens(), 3);
        assert_eq!(t.nrels(), 2);
        assert_eq!(t.relations()[1].lhs(), &[2]); // y
        assert_eq!(t.relations()[1].rhs(), &[0, 1]);
        // delete generator 2 (occurs in exactly one relation): drops that relation
        let u = t.delete_generator(2);
        assert_eq!(u.ngens(), 2);
        assert_eq!(u.nrels(), 1);
        assert_eq!(u.relations()[0].lhs(), &[0, 1]);
    }

    #[test]
    fn direct_and_free_products() {
        let r = FpSemigroup::semigroup(1, vec![Relation::new(vec![0, 0], vec![0])]);
        let s = FpSemigroup::semigroup(1, vec![Relation::new(vec![0, 0, 0], vec![0])]);
        let dp = r.direct_product(&s);
        assert_eq!(dp.ngens(), 2);
        // relations: r's (x0^2=x0), s's shifted (x1^3=x1), and commuting x0 x1 = x1 x0
        assert_eq!(dp.nrels(), 3);
        assert!(dp
            .relations()
            .iter()
            .any(|rel| rel.lhs() == [0, 1] && rel.rhs() == [1, 0]));
        let fp = r.free_product(&s);
        assert_eq!(fp.ngens(), 2);
        assert_eq!(fp.nrels(), 2); // no commuting relations
    }

    #[test]
    fn sub_and_ideal_descriptors() {
        let s = FpSemigroup::semigroup(2, vec![]);
        let sub = s.sub(&[s.element_word(&[0]), s.element_word(&[0, 1])]);
        assert_eq!(sub.kind(), SubKind::Subsemigroup);
        assert_eq!(sub.generators().len(), 2);
        let id = s.ideal(&[s.element_word(&[1])]);
        assert_eq!(id.kind(), SubKind::TwoSidedIdeal);
    }
}

impl FpSemigroup {
    /// Helper: a word `S ! [i1, …]` in this presentation.
    pub fn element_word(&self, indices: &[usize]) -> FreeSemigroupElement {
        FreeSemigroupElement::new(indices.to_vec())
    }
}
