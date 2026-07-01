//! Groups defined by rewrite systems (MAGMA handbook chapter 74, `GrpRWS`).
//!
//! A rewrite group is a finitely presented group whose word problem is decided
//! by a confluent rewriting system (when Knuth–Bendix completion succeeds). The
//! group presentation `⟨ g_1, …, g_n | R ⟩` is turned into a *monoid*
//! presentation on the `2n` letters `g_1, g_1⁻¹, …, g_n, g_n⁻¹` with the trivial
//! inverse relations `g_i g_i⁻¹ = g_i⁻¹ g_i = 1`, and completion is run on that
//! (§74.2.1). The resulting [`ReductionAutomaton`] provides normal forms, so the
//! group elements form a genuine `rustmath_core::Group`.
//!
//! Words are represented internally over the `2n`-letter monoid alphabet, where
//! letter `2i` is `g_{i+1}` and letter `2i+1` is `g_{i+1}⁻¹`; the involution is
//! `letter XOR 1`. MAGMA's signed generator sequences `G ! [i1, …, is]` (with
//! `i_j ∈ [−n, n] \ {0}`) map onto this alphabet.

use crate::knuth_bendix::{KbLimits, RewritingSystem};
use crate::reduction_automaton::{GrowthFunction, ReductionAutomaton};
use crate::word_ordering::WordOrdering;
use rustmath_core::{Group, Magma, Monoid, Semigroup};
use rustmath_integers::Integer;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Convert a signed generator index `s` (`±(i+1)`) to a monoid-alphabet letter.
fn signed_to_letter(s: i64) -> usize {
    let i = (s.unsigned_abs() as usize) - 1;
    if s > 0 {
        2 * i
    } else {
        2 * i + 1
    }
}

/// Convert a monoid-alphabet letter back to a signed generator index.
fn letter_to_signed(l: usize) -> i64 {
    let i = (l / 2) as i64 + 1;
    if l % 2 == 0 {
        i
    } else {
        -i
    }
}

/// A group defined by a rewrite system (`GrpRWS`).
#[derive(Debug, Clone)]
pub struct RwsGroup {
    num_gens: usize,
    machine: Rc<ReductionAutomaton>,
    /// The user relators as signed generator words `w` (meaning `w = 1`).
    relators: Vec<Vec<i64>>,
}

impl RwsGroup {
    /// Construct a rewrite group from a group presentation.
    ///
    /// `num_gens` is the number of group generators `n`; `relators` are signed
    /// generator words `w` (each meaning `w = 1`). `ordering` is the word
    /// ordering used by completion (§74.2.2). Knuth–Bendix completion is run
    /// within `limits`; use [`RwsGroup::is_confluent`] to check whether a
    /// confluent presentation was obtained.
    pub fn new(
        num_gens: usize,
        relators: &[Vec<i64>],
        ordering: WordOrdering,
        limits: &KbLimits,
    ) -> Self {
        let alphabet = 2 * num_gens;
        let mut relations: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        // trivial inverse relations g_i g_i^-1 = 1, g_i^-1 g_i = 1
        for i in 0..num_gens {
            relations.push((vec![2 * i, 2 * i + 1], vec![]));
            relations.push((vec![2 * i + 1, 2 * i], vec![]));
        }
        // group relators w = 1
        for w in relators {
            let letters: Vec<usize> = w.iter().map(|&s| signed_to_letter(s)).collect();
            relations.push((letters, vec![]));
        }
        let mut sys = RewritingSystem::from_relations(alphabet, ordering, &relations);
        sys.complete(limits);
        let inverse_map: Vec<usize> = (0..alphabet).map(|l| l ^ 1).collect();
        let machine = Rc::new(ReductionAutomaton::from_system(&sys, Some(inverse_map)));
        RwsGroup {
            num_gens,
            machine,
            relators: relators.to_vec(),
        }
    }

    /// Construct with the default Knuth–Bendix limits.
    pub fn from_presentation(
        num_gens: usize,
        relators: &[Vec<i64>],
        ordering: WordOrdering,
    ) -> Self {
        RwsGroup::new(num_gens, relators, ordering, &KbLimits::default())
    }

    /// `NumberOfGenerators` / `Ngens`.
    pub fn ngens(&self) -> usize {
        self.num_gens
    }

    /// The reduction machine underlying the group.
    pub fn machine(&self) -> &Rc<ReductionAutomaton> {
        &self.machine
    }

    /// Whether completion produced a confluent presentation (`IsConfluent`).
    pub fn is_confluent(&self) -> bool {
        self.machine.is_confluent()
    }

    /// The word ordering used (`Ordering(G)`).
    pub fn ordering_name(&self) -> &'static str {
        self.machine.ordering().name()
    }

    /// The `i`-th defining generator (`G . i`), `1 ≤ i ≤ n`. Signed `i`
    /// (`−n ≤ i ≤ −1`) yields the corresponding inverse generator.
    pub fn generator(&self, i: i64) -> RwsGroupElement {
        self.element(&[i])
    }

    /// A sequence of the defining generators (`Generators(G)`).
    pub fn generators(&self) -> Vec<RwsGroupElement> {
        (1..=self.num_gens as i64).map(|i| self.generator(i)).collect()
    }

    /// The identity word (`Identity(G)` / `Id(G)` / `G ! 1`).
    pub fn identity(&self) -> RwsGroupElement {
        RwsGroupElement {
            word: Vec::new(),
            machine: Some(self.machine.clone()),
        }
    }

    /// Construct `G ! [i1, …, is]`: the reduced word
    /// `G.|i1|^{±1} * … * G.|is|^{±1}`.
    pub fn element(&self, signed: &[i64]) -> RwsGroupElement {
        let letters: Vec<usize> = signed.iter().map(|&s| signed_to_letter(s)).collect();
        let word = self.machine.reduce(&letters);
        RwsGroupElement {
            word,
            machine: Some(self.machine.clone()),
        }
    }

    /// The defining relations (`Relations(G)`) as signed generator words `w`
    /// (each interpreted as `w = 1`).
    pub fn relations(&self) -> &[Vec<i64>] {
        &self.relators
    }

    /// `NumberOfRelations` / `Nrels`.
    pub fn nrels(&self) -> usize {
        self.relators.len()
    }

    /// The order of the group (`Order(G)` / `#G`); `None` means infinite.
    pub fn order(&self) -> Option<Integer> {
        self.machine.order()
    }

    /// `IsFinite(G)` with the order when finite.
    pub fn is_finite(&self) -> (bool, Option<Integer>) {
        self.machine.is_finite()
    }

    /// The rational growth function (`GrowthFunction`).
    pub fn growth_function(&self) -> GrowthFunction {
        self.machine.growth_function()
    }

    /// Enumerate reduced words of length in `[a, b]` (`Set`/`Seq`).
    pub fn enumerate(&self, a: usize, b: usize, bfs: bool) -> Vec<RwsGroupElement> {
        self.machine
            .enumerate(a, b, bfs)
            .into_iter()
            .map(|word| RwsGroupElement {
                word,
                machine: Some(self.machine.clone()),
            })
            .collect()
    }
}

/// An element (reduced word) of a rewrite group (`GrpRWSElt`).
#[derive(Debug, Clone)]
pub struct RwsGroupElement {
    word: Vec<usize>,
    machine: Option<Rc<ReductionAutomaton>>,
}

impl RwsGroupElement {
    /// The normal-form word over the `2n`-letter monoid alphabet.
    pub fn letters(&self) -> &[usize] {
        &self.word
    }

    /// The length of the word (`#u`).
    pub fn len(&self) -> usize {
        self.word.len()
    }

    /// Whether the word is the identity (empty).
    pub fn is_empty(&self) -> bool {
        self.word.is_empty()
    }

    /// `ElementToSequence` / `Eltseq`: signed generator indices of the word.
    pub fn eltseq(&self) -> Vec<i64> {
        self.word.iter().map(|&l| letter_to_signed(l)).collect()
    }

    fn parent(&self, other: &RwsGroupElement) -> Option<Rc<ReductionAutomaton>> {
        self.machine.clone().or_else(|| other.machine.clone())
    }

    fn reduced(word: Vec<usize>, machine: Option<Rc<ReductionAutomaton>>) -> Self {
        let word = match &machine {
            Some(m) => m.reduce(&word),
            None => word,
        };
        RwsGroupElement { word, machine }
    }

    /// Product `u / v = u * v⁻¹` (§74.4.2).
    pub fn div(&self, other: &RwsGroupElement) -> RwsGroupElement {
        self.op(&other.inverse())
    }

    /// `n`-th power `u ^ n` (negative `n` powers the inverse).
    pub fn pow(&self, n: i64) -> RwsGroupElement {
        let base = if n < 0 { self.inverse() } else { self.clone() };
        let mut acc = RwsGroupElement {
            word: Vec::new(),
            machine: self.machine.clone(),
        };
        for _ in 0..n.unsigned_abs() {
            acc = acc.op(&base);
        }
        acc
    }

    /// Conjugate `u ^ v = v⁻¹ * u * v`.
    pub fn conjugate(&self, v: &RwsGroupElement) -> RwsGroupElement {
        v.inverse().op(self).op(v)
    }

    /// Commutator `(u, v) = u⁻¹ v⁻¹ u v`.
    pub fn commutator(&self, v: &RwsGroupElement) -> RwsGroupElement {
        self.inverse().op(&v.inverse()).op(self).op(v)
    }
}

impl PartialEq for RwsGroupElement {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}
impl Eq for RwsGroupElement {}
impl Hash for RwsGroupElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

impl Magma for RwsGroupElement {
    fn op(&self, other: &Self) -> Self {
        let machine = self.parent(other);
        let mut word = self.word.clone();
        word.extend_from_slice(&other.word);
        RwsGroupElement::reduced(word, machine)
    }
}

impl Semigroup for RwsGroupElement {}

impl Monoid for RwsGroupElement {
    fn identity() -> Self {
        RwsGroupElement {
            word: Vec::new(),
            machine: None,
        }
    }

    fn is_identity(&self) -> bool {
        self.word.is_empty()
    }
}

impl Group for RwsGroupElement {
    fn inverse(&self) -> Self {
        let inv_word: Vec<usize> = match &self.machine {
            Some(m) => {
                let map = m.inverse_map().expect("group machine has an inverse map");
                self.word.iter().rev().map(|&l| map[l]).collect()
            }
            None => self.word.iter().rev().map(|&l| l ^ 1).collect(),
        };
        RwsGroupElement::reduced(inv_word, self.machine.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Von Dyck (2,3,5) group ≅ A5: ⟨a,b | a^2, b^3, (ab)^5⟩, order 60.
    fn a5() -> RwsGroup {
        RwsGroup::from_presentation(
            2,
            &[vec![1, 1], vec![2, 2, 2], vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]],
            WordOrdering::ShortLex,
        )
    }

    #[test]
    fn a5_is_confluent_order_60() {
        let g = a5();
        assert!(g.is_confluent());
        assert_eq!(g.order(), Some(Integer::from(60u64)));
        let (fin, ord) = g.is_finite();
        assert!(fin);
        assert_eq!(ord, Some(Integer::from(60u64)));
        assert_eq!(g.ngens(), 2);
        assert_eq!(g.ordering_name(), "ShortLex");
    }

    #[test]
    fn a5_group_axioms() {
        let g = a5();
        let a = g.generator(1);
        let b = g.generator(2);
        // a^2 = 1, b^3 = 1
        assert!(a.op(&a).is_identity());
        assert!(b.pow(3).is_identity());
        // inverse
        let ab = a.op(&b);
        assert!(ab.op(&ab.inverse()).is_identity());
        assert!(ab.inverse().op(&ab).is_identity());
        // identity element
        let id = g.identity();
        assert_eq!(a.op(&id), a);
        assert_eq!(id.op(&a), a);
    }

    #[test]
    fn word_construction_and_eltseq() {
        let g = a5();
        // G ! [1, 2] = a*b
        let w = g.element(&[1, 2]);
        assert_eq!(w, g.generator(1).op(&g.generator(2)));
        // a*b is not the identity; a*a is
        assert!(!w.is_identity());
        assert!(g.element(&[1, -1]).is_identity()); // a a^-1 = 1
        // eltseq round-trips through the generator alphabet
        let seq = w.eltseq();
        assert_eq!(g.element(&seq), w);
    }

    #[test]
    fn commutator_and_conjugate() {
        let g = a5();
        let a = g.generator(1);
        let b = g.generator(2);
        // (a,b) = a^-1 b^-1 a b
        let comm = a.commutator(&b);
        let expected = a.inverse().op(&b.inverse()).op(&a).op(&b);
        assert_eq!(comm, expected);
        // conjugate a^b = b^-1 a b
        let conj = a.conjugate(&b);
        assert_eq!(conj, b.inverse().op(&a).op(&b));
    }

    #[test]
    fn infinite_group_reports_infinite_order() {
        // ⟨a,b | ⟩ : free group of rank 2 is infinite.
        let g = RwsGroup::from_presentation(2, &[], WordOrdering::ShortLex);
        assert_eq!(g.order(), None);
        let (fin, _) = g.is_finite();
        assert!(!fin);
    }
}
