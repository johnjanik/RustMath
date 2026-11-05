//! Monoids given by rewrite systems (MAGMA handbook chapter 78, `MonRWS`).
//!
//! A rewrite monoid is a finitely presented monoid whose word problem is decided
//! by a confluent rewriting system. It is built by applying Knuth–Bendix
//! completion (the shared engine in [`rustmath_automata`]) to a free-monoid
//! presentation `⟨ x_1, …, x_n | relations ⟩`. Unlike a rewrite *group* there are
//! no inverses, so elements implement the `rustmath_core`
//! `Magma → Semigroup → Monoid` tower but not `Group`.
//!
//! Elements are reduced words over the `n`-letter generator alphabet; the empty
//! word is the identity. `Order` returns `Option<Integer>` (`None` = infinite),
//! never forcing a finiteness decision.
//!
//! Generator indices are 0-based here (MAGMA numbers them from 1).

use rustmath_automata::{GrowthFunction, KbLimits, ReductionAutomaton, RewritingSystem, WordOrdering};
use rustmath_core::{Magma, Monoid, Semigroup};
use rustmath_integers::Integer;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// A monoid defined by a rewrite system (`MonRWS`).
#[derive(Debug, Clone)]
pub struct RwsMonoid {
    num_generators: usize,
    machine: Rc<ReductionAutomaton>,
    relations: Vec<(Vec<usize>, Vec<usize>)>,
}

impl RwsMonoid {
    /// `RWSMonoid(Q : parameters)`: run Knuth–Bendix completion on the free
    /// monoid on `n` generators modulo `relations`, using `ordering` and
    /// `limits`. Use [`RwsMonoid::is_confluent`] to check whether the result is
    /// confluent.
    pub fn new(
        num_generators: usize,
        relations: &[(Vec<usize>, Vec<usize>)],
        ordering: WordOrdering,
        limits: &KbLimits,
    ) -> Self {
        let mut sys = RewritingSystem::from_relations(num_generators, ordering, relations);
        sys.complete(limits);
        let machine = Rc::new(ReductionAutomaton::from_system(&sys, None));
        RwsMonoid {
            num_generators,
            machine,
            relations: relations.to_vec(),
        }
    }

    /// Construct with default Knuth–Bendix limits.
    pub fn from_presentation(
        num_generators: usize,
        relations: &[(Vec<usize>, Vec<usize>)],
        ordering: WordOrdering,
    ) -> Self {
        RwsMonoid::new(num_generators, relations, ordering, &KbLimits::default())
    }

    /// The reduction machine underlying the monoid.
    pub fn machine(&self) -> &Rc<ReductionAutomaton> {
        &self.machine
    }

    /// `NumberOfGenerators` / `Ngens`.
    pub fn ngens(&self) -> usize {
        self.num_generators
    }

    /// `IsConfluent(M)`.
    pub fn is_confluent(&self) -> bool {
        self.machine.is_confluent()
    }

    /// The word ordering used (`Ordering(M)`).
    pub fn ordering_name(&self) -> &'static str {
        self.machine.ordering().name()
    }

    /// The `i`-th defining generator `M.i` (0-based).
    pub fn gen(&self, i: usize) -> Option<RwsMonoidElement> {
        if i < self.num_generators {
            Some(self.element(&[i]))
        } else {
            None
        }
    }

    /// `Generators(M)`.
    pub fn generators(&self) -> Vec<RwsMonoidElement> {
        (0..self.num_generators)
            .filter_map(|i| self.gen(i))
            .collect()
    }

    /// The identity word (`Identity(M)` / `Id(M)` / `M ! 1`).
    pub fn identity(&self) -> RwsMonoidElement {
        RwsMonoidElement {
            word: Vec::new(),
            machine: Some(self.machine.clone()),
        }
    }

    /// `M ! [i1, …, is]`: the reduced word `M.i1 * … * M.is` (0-based indices).
    pub fn element(&self, indices: &[usize]) -> RwsMonoidElement {
        let word = self.machine.reduce(indices);
        RwsMonoidElement {
            word,
            machine: Some(self.machine.clone()),
        }
    }

    /// `Relations(M)`: the defining relations as word pairs.
    pub fn relations(&self) -> &[(Vec<usize>, Vec<usize>)] {
        &self.relations
    }

    /// `NumberOfRelations` / `Nrels`.
    pub fn nrels(&self) -> usize {
        self.relations.len()
    }

    /// The confluent rewrite relations as word pairs (`Relations` of the
    /// completed system, suitable for conversion back to an fp-monoid).
    pub fn confluent_relations(&self) -> Vec<(Vec<usize>, Vec<usize>)> {
        self.machine
            .rules()
            .iter()
            .map(|r| (r.lhs.clone(), r.rhs.clone()))
            .collect()
    }

    /// `Order(M)` / `#M`; `None` means infinite.
    pub fn order(&self) -> Option<Integer> {
        self.machine.order()
    }

    /// `IsFinite(M)` with the order when finite.
    pub fn is_finite(&self) -> (bool, Option<Integer>) {
        self.machine.is_finite()
    }

    /// `GrowthFunction`.
    pub fn growth_function(&self) -> GrowthFunction {
        self.machine.growth_function()
    }

    /// Enumerate reduced words of length in `[a, b]` (`Set`/`Seq`).
    pub fn enumerate(&self, a: usize, b: usize, bfs: bool) -> Vec<RwsMonoidElement> {
        self.machine
            .enumerate(a, b, bfs)
            .into_iter()
            .map(|word| RwsMonoidElement {
                word,
                machine: Some(self.machine.clone()),
            })
            .collect()
    }
}

/// An element (reduced word) of a rewrite monoid (`MonRWSElt`).
#[derive(Debug, Clone)]
pub struct RwsMonoidElement {
    word: Vec<usize>,
    machine: Option<Rc<ReductionAutomaton>>,
}

impl RwsMonoidElement {
    /// The normal-form word (0-based generator indices).
    pub fn word(&self) -> &[usize] {
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
    pub fn eltseq(&self) -> Vec<usize> {
        self.word.clone()
    }

    /// `u ^ n` for `n ≥ 0`.
    pub fn pow(&self, n: usize) -> RwsMonoidElement {
        let mut acc = RwsMonoidElement {
            word: Vec::new(),
            machine: self.machine.clone(),
        };
        for _ in 0..n {
            acc = acc.op(self);
        }
        acc
    }

    fn parent(&self, other: &RwsMonoidElement) -> Option<Rc<ReductionAutomaton>> {
        self.machine.clone().or_else(|| other.machine.clone())
    }
}

impl PartialEq for RwsMonoidElement {
    fn eq(&self, other: &Self) -> bool {
        self.word == other.word
    }
}
impl Eq for RwsMonoidElement {}
impl Hash for RwsMonoidElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.word.hash(state);
    }
}

impl Magma for RwsMonoidElement {
    fn op(&self, other: &Self) -> Self {
        let machine = self.parent(other);
        let mut word = self.word.clone();
        word.extend_from_slice(&other.word);
        let word = match &machine {
            Some(m) => m.reduce(&word),
            None => word,
        };
        RwsMonoidElement { word, machine }
    }
}

impl Semigroup for RwsMonoidElement {}

impl Monoid for RwsMonoidElement {
    fn identity() -> Self {
        RwsMonoidElement {
            word: Vec::new(),
            machine: None,
        }
    }

    fn is_identity(&self) -> bool {
        self.word.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // MAGMA H78E1: alternating group A4 as a rewrite monoid.
    // Monoid presentation ⟨a,b | a^2, b^3, (ab)^3⟩ (a,b as monoid gens with the
    // relations forcing torsion). Under ShortLex this completes confluently.
    #[test]
    fn a4_rewrite_monoid_order_12() {
        // Use a group-style monoid presentation with explicit inverses so the
        // monoid really is the finite group A4 (order 12).
        // gens: a=0, A=1 (=a^-1), b=2, B=3 (=b^-1)
        let rels: Vec<(Vec<usize>, Vec<usize>)> = vec![
            (vec![0, 1], vec![]),
            (vec![1, 0], vec![]),
            (vec![2, 3], vec![]),
            (vec![3, 2], vec![]),
            (vec![0, 0], vec![]),          // a^2 = 1
            (vec![2, 2, 2], vec![]),       // b^3 = 1
            (vec![0, 2, 0, 2, 0, 2], vec![]), // (ab)^3 = 1
        ];
        let m = RwsMonoid::from_presentation(4, &rels, WordOrdering::ShortLex);
        assert!(m.is_confluent());
        assert_eq!(m.order(), Some(Integer::from(12u64)));
        let (fin, ord) = m.is_finite();
        assert!(fin);
        assert_eq!(ord, Some(Integer::from(12u64)));
    }

    #[test]
    fn free_commutative_monoid_is_a_core_monoid() {
        // ⟨a,b | ba = ab⟩ : normal forms a^i b^j, infinite order.
        let m = RwsMonoid::from_presentation(
            2,
            &[(vec![1, 0], vec![0, 1])],
            WordOrdering::ShortLex,
        );
        assert!(m.is_confluent());
        assert_eq!(m.order(), None);
        let a = m.gen(0).unwrap();
        let b = m.gen(1).unwrap();
        // associativity + reduction: b*a*b*a -> a a b b
        let w = b.op(&a).op(&b).op(&a);
        assert_eq!(w.word(), &[0, 0, 1, 1]);
        // identity laws
        let id = m.identity();
        assert_eq!(a.op(&id), a);
        assert_eq!(id.op(&b), b);
        assert!(id.is_identity());
        // eq is decided by normal form: ab == ba
        assert_eq!(a.op(&b), b.op(&a));
        assert_ne!(a.op(&a), a.op(&b));
    }

    #[test]
    fn h78e10_fibonacci_style_reduction() {
        // A finite commutative example: ⟨a,b | a^2, b^2, ab=ba⟩ -> order 4,
        // normal forms 1, a, b, ab.
        let m = RwsMonoid::from_presentation(
            2,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![1, 0], vec![0, 1]),
            ],
            WordOrdering::ShortLex,
        );
        assert_eq!(m.order(), Some(Integer::from(4u64)));
        // IsIdentity(a^0) is true; b^2 reduces to identity
        assert!(m.gen(0).unwrap().pow(0).is_identity());
        assert!(m.gen(1).unwrap().pow(2).is_identity());
        // enumerate all 4 elements
        let all = m.enumerate(0, 2, true);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn confluent_relations_are_exposed() {
        let m = RwsMonoid::from_presentation(
            2,
            &[(vec![1, 0], vec![0, 1])],
            WordOrdering::ShortLex,
        );
        let rels = m.confluent_relations();
        assert!(rels.iter().any(|(l, r)| l == &[1, 0] && r == &[0, 1]));
    }

    #[test]
    fn recursive_ordering_submonoid() {
        // MAGMA H78E3 flavour: a submonoid with the Recursive ordering that is
        // infinite. ⟨a,b | ba = a^2 b⟩ under Recursive completes.
        let m = RwsMonoid::from_presentation(
            2,
            &[(vec![1, 0], vec![0, 0, 1])],
            WordOrdering::Recursive,
        );
        assert!(m.is_confluent());
        assert_eq!(m.order(), None); // infinite
    }
}
