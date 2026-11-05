//! Automatic groups (MAGMA handbook chapter 75, `GrpAtc`).
//!
//! An automatic group is a finitely presented group whose word problem and word
//! enumeration are decided by finite automata, with words compared under the
//! short-lex ordering. Following MAGMA (§75.2.1), the group presentation is
//! converted to a monoid presentation on the generators-and-inverses alphabet
//! and Knuth–Bendix completion is run under the short-lex ordering.
//!
//! When completion yields a finite confluent system we obtain the short-lex
//! automatic structure: the **word acceptor** (a DFA recognising normal-form
//! words) and a **word multiplier** realised by the reduction machine. Both are
//! provided here via [`ReductionAutomaton`], and elements form a
//! `rustmath_core::Group`.
//!
//! Deferred (documented gap): the explicit *word-difference automata*
//! construction and its independent verification step. `AutomaticGroup` here
//! derives the automatic structure from the confluent rewriting system rather
//! than from word-difference machines; the observable results (`WordAcceptor`,
//! `Order`, `GrowthFunction`, multiplication) are real, but
//! `WordDifferenceAutomaton` is not built.

use crate::knuth_bendix::{KbLimits, RewritingSystem};
use crate::reduction_automaton::{GrowthFunction, ReductionAutomaton};
use crate::word_ordering::WordOrdering;
use crate::DFA;
use rustmath_core::{Group, Magma, Monoid, Semigroup};
use rustmath_integers::Integer;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

fn signed_to_letter(s: i64) -> usize {
    let i = (s.unsigned_abs() as usize) - 1;
    if s > 0 {
        2 * i
    } else {
        2 * i + 1
    }
}

fn letter_to_signed(l: usize) -> i64 {
    let i = (l / 2) as i64 + 1;
    if l % 2 == 0 {
        i
    } else {
        -i
    }
}

/// A short-lex automatic group (`GrpAtc`).
#[derive(Debug, Clone)]
pub struct AutomaticGroup {
    num_gens: usize,
    machine: Rc<ReductionAutomaton>,
    relators: Vec<Vec<i64>>,
}

impl AutomaticGroup {
    /// Attempt to construct an automatic structure for the group
    /// `⟨ g_1, …, g_n | relators ⟩` (`AutomaticGroup` / `IsAutomaticGroup`).
    ///
    /// Returns `Some(G)` when short-lex Knuth–Bendix completion succeeds within
    /// `limits` (a confluent system, hence a short-lex automatic structure), and
    /// `None` on failure — matching MAGMA's `IsAutomaticGroup` returning
    /// `(true, G)` or `false`.
    pub fn try_new(num_gens: usize, relators: &[Vec<i64>], limits: &KbLimits) -> Option<Self> {
        let alphabet = 2 * num_gens;
        let mut relations: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for i in 0..num_gens {
            relations.push((vec![2 * i, 2 * i + 1], vec![]));
            relations.push((vec![2 * i + 1, 2 * i], vec![]));
        }
        for w in relators {
            let letters: Vec<usize> = w.iter().map(|&s| signed_to_letter(s)).collect();
            relations.push((letters, vec![]));
        }
        let mut sys = RewritingSystem::from_relations(alphabet, WordOrdering::ShortLex, &relations);
        if !sys.complete(limits) {
            return None;
        }
        let inverse_map: Vec<usize> = (0..alphabet).map(|l| l ^ 1).collect();
        let machine = Rc::new(ReductionAutomaton::from_system(&sys, Some(inverse_map)));
        Some(AutomaticGroup {
            num_gens,
            machine,
            relators: relators.to_vec(),
        })
    }

    /// Convenience wrapper using default limits.
    pub fn from_presentation(num_gens: usize, relators: &[Vec<i64>]) -> Option<Self> {
        AutomaticGroup::try_new(num_gens, relators, &KbLimits::default())
    }

    /// `NumberOfGenerators` / `Ngens`.
    pub fn ngens(&self) -> usize {
        self.num_gens
    }

    /// The reduction machine (word multiplier).
    pub fn machine(&self) -> &Rc<ReductionAutomaton> {
        &self.machine
    }

    /// The word-acceptor automaton as a DFA (`WordAcceptor`).
    pub fn word_acceptor(&self) -> DFA<usize, usize> {
        self.machine.word_acceptor_dfa()
    }

    /// The number of states of the word acceptor and its alphabet size
    /// (`WordAcceptorSize`).
    pub fn word_acceptor_size(&self) -> (usize, usize) {
        (self.machine.word_acceptor_size(), 2 * self.num_gens)
    }

    /// The `i`-th defining generator (`G . i`).
    pub fn generator(&self, i: i64) -> AutomaticGroupElement {
        self.element(&[i])
    }

    /// The defining generators (`Generators(G)`).
    pub fn generators(&self) -> Vec<AutomaticGroupElement> {
        (1..=self.num_gens as i64).map(|i| self.generator(i)).collect()
    }

    /// The identity element (`Identity(G)` / `Id(G)`).
    pub fn identity(&self) -> AutomaticGroupElement {
        AutomaticGroupElement {
            word: Vec::new(),
            machine: Some(self.machine.clone()),
        }
    }

    /// Construct `G ! [i1, …, is]`, reduced to normal form.
    pub fn element(&self, signed: &[i64]) -> AutomaticGroupElement {
        let letters: Vec<usize> = signed.iter().map(|&s| signed_to_letter(s)).collect();
        let word = self.machine.reduce(&letters);
        AutomaticGroupElement {
            word,
            machine: Some(self.machine.clone()),
        }
    }

    /// The defining relations as signed generator words.
    pub fn relations(&self) -> &[Vec<i64>] {
        &self.relators
    }

    /// The order (`Order(G)` / `#G`); `None` means infinite.
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
    pub fn enumerate(&self, a: usize, b: usize, bfs: bool) -> Vec<AutomaticGroupElement> {
        self.machine
            .enumerate(a, b, bfs)
            .into_iter()
            .map(|word| AutomaticGroupElement {
                word,
                machine: Some(self.machine.clone()),
            })
            .collect()
    }
}

/// An element (reduced word) of an automatic group (`GrpAtcElt`).
#[derive(Debug, Clone)]
pub struct AutomaticGroupElement {
    word: Vec<usize>,
    machine: Option<Rc<ReductionAutomaton>>,
}

impl AutomaticGroupElement {
    /// The normal-form word over the generators-and-inverses alphabet.
    pub fn letters(&self) -> &[usize] {
        &self.word
    }

    /// The length of the word (`#u`).
    pub fn len(&self) -> usize {
        self.word.len()
    }

    /// Whether the word is empty (the identity).
    pub fn is_empty(&self) -> bool {
        self.word.is_empty()
    }

    /// `ElementToSequence` / `Eltseq`.
    pub fn eltseq(&self) -> Vec<i64> {
        self.word.iter().map(|&l| letter_to_signed(l)).collect()
    }

    fn parent(&self, other: &AutomaticGroupElement) -> Option<Rc<ReductionAutomaton>> {
        self.machine.clone().or_else(|| other.machine.clone())
    }

    fn reduced(word: Vec<usize>, machine: Option<Rc<ReductionAutomaton>>) -> Self {
        let word = match &machine {
            Some(m) => m.reduce(&word),
            None => word,
        };
        AutomaticGroupElement { word, machine }
    }

    /// `u / v = u * v⁻¹`.
    pub fn div(&self, other: &AutomaticGroupElement) -> AutomaticGroupElement {
        self.op(&other.inverse())
    }

    /// `n`-th power (negative `n` powers the inverse).
    pub fn pow(&self, n: i64) -> AutomaticGroupElement {
        let base = if n < 0 { self.inverse() } else { self.clone() };
        let mut acc = AutomaticGroupElement {
            word: Vec::new(),
            machine: self.machine.clone(),
        };
        for _ in 0..n.unsigned_abs() {
            acc = acc.op(&base);
        }
        acc
    }

    /// Conjugate `u ^ v = v⁻¹ * u * v`.
    pub fn conjugate(&self, v: &AutomaticGroupElement) -> AutomaticGroupElement {
        v.inverse().op(self).op(v)
    }

    /// Commutator `(u, v) = u⁻¹ v⁻¹ u v`.
    pub fn commutator(&self, v: &AutomaticGroupElement) -> AutomaticGroupElement {
        self.inverse().op(&v.inverse()).op(self).op(v)
    }
}

impl PartialEq for AutomaticGroupElement {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}
impl Eq for AutomaticGroupElement {}
impl Hash for AutomaticGroupElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

impl Magma for AutomaticGroupElement {
    fn op(&self, other: &Self) -> Self {
        let machine = self.parent(other);
        let mut word = self.word.clone();
        word.extend_from_slice(&other.word);
        AutomaticGroupElement::reduced(word, machine)
    }
}

impl Semigroup for AutomaticGroupElement {}

impl Monoid for AutomaticGroupElement {
    fn identity() -> Self {
        AutomaticGroupElement {
            word: Vec::new(),
            machine: None,
        }
    }

    fn is_identity(&self) -> bool {
        self.word.is_empty()
    }
}

impl Group for AutomaticGroupElement {
    fn inverse(&self) -> Self {
        let inv_word: Vec<usize> = match &self.machine {
            Some(m) => {
                let map = m.inverse_map().expect("group machine has an inverse map");
                self.word.iter().rev().map(|&l| map[l]).collect()
            }
            None => self.word.iter().rev().map(|&l| l ^ 1).collect(),
        };
        AutomaticGroupElement::reduced(inv_word, self.machine.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn von_dyck_235_is_automatic_and_a5() {
        // Von Dyck (2,3,5) group ≅ A5 (MAGMA H75E4).
        let g = AutomaticGroup::from_presentation(
            2,
            &[vec![1, 1], vec![2, 2, 2], vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]],
        )
        .expect("should be automatic");
        assert_eq!(g.order(), Some(Integer::from(60u64)));
        assert_eq!(g.ngens(), 2);
        let (states, alph) = g.word_acceptor_size();
        assert!(states >= 1);
        assert_eq!(alph, 4);
    }

    #[test]
    fn word_acceptor_recognises_normal_forms() {
        let g = AutomaticGroup::from_presentation(2, &[vec![1, 1], vec![2, 2]])
            .expect("automatic");
        let dfa = g.word_acceptor();
        // a a^-1 letters are 0 and 1; a^2 (letters [0,0]) is reducible.
        assert!(!dfa.accepts(&[0, 0]));
    }

    #[test]
    fn infinite_dihedral_growth_and_order() {
        // ⟨a,b | a^2, b^2⟩ = infinite dihedral (MAGMA H75E5/H75E10 flavour).
        let g = AutomaticGroup::from_presentation(2, &[vec![1, 1], vec![2, 2]])
            .expect("automatic");
        assert_eq!(g.order(), None);
        let gf = g.growth_function();
        // (1 + x) / (1 - x)
        assert_eq!(gf.numerator, vec![Integer::from(1u64), Integer::from(1u64)]);
        assert_eq!(
            gf.denominator,
            vec![Integer::from(1u64), Integer::from(-1i64)]
        );
    }

    #[test]
    fn group_arithmetic() {
        let g = AutomaticGroup::from_presentation(
            2,
            &[vec![1, 1], vec![2, 2, 2], vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2]],
        )
        .unwrap();
        let a = g.generator(1);
        let b = g.generator(2);
        assert!(a.pow(2).is_identity());
        assert!(b.pow(3).is_identity());
        let ab = a.op(&b);
        assert!(ab.div(&ab).is_identity());
        assert_eq!(g.element(&[1, 2]), ab);
    }
}
