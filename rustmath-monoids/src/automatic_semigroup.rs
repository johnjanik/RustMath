//! # Automatic Semigroups
//!
//! This module provides automatic semigroups and monoids.
//!
//! An automatic semigroup is a semigroup with a regular language of normal forms
//! and a finite-state automaton for computing the product. Products are computed
//! by concatenation followed by reduction to normal form using the
//! Knuth–Bendix reduction machine from [`rustmath_automata`] (MAGMA handbook
//! chapters 75/78).

use rustmath_automata::{KbLimits, ReductionAutomaton, RewritingSystem, WordOrdering};
use std::collections::HashMap;
use std::rc::Rc;

/// An element of an automatic semigroup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Element {
    /// Normal form representation
    normal_form: Vec<usize>,
}

impl Element {
    /// Create a new element
    pub fn new(normal_form: Vec<usize>) -> Self {
        Element { normal_form }
    }

    /// Get the normal form
    pub fn normal_form(&self) -> &[usize] {
        &self.normal_form
    }
}

/// An automatic semigroup
#[derive(Debug, Clone)]
pub struct AutomaticSemigroup {
    /// Number of generators
    num_generators: usize,
    /// Multiplication table (if finite)
    #[allow(dead_code)]
    mult_table: Option<HashMap<(usize, usize), usize>>,
    /// Reduction machine for computing normal forms of products. When absent,
    /// the semigroup is free and products are plain concatenations.
    machine: Option<Rc<ReductionAutomaton>>,
}

impl AutomaticSemigroup {
    /// Create a new (free) automatic semigroup on `num_generators` generators.
    pub fn new(num_generators: usize) -> Self {
        AutomaticSemigroup {
            num_generators,
            mult_table: None,
            machine: None,
        }
    }

    /// Create an automatic semigroup from a confluent (or best-effort) rewrite
    /// presentation: `num_generators` generators subject to `relations`, using
    /// the given word `ordering`. Knuth–Bendix completion builds the reduction
    /// machine used by [`AutomaticSemigroup::mul`].
    pub fn from_relations(
        num_generators: usize,
        relations: &[(Vec<usize>, Vec<usize>)],
        ordering: WordOrdering,
    ) -> Self {
        let mut sys = RewritingSystem::from_relations(num_generators, ordering, relations);
        sys.complete(&KbLimits::default());
        let machine = Rc::new(ReductionAutomaton::from_system(&sys, None));
        AutomaticSemigroup {
            num_generators,
            mult_table: None,
            machine: Some(machine),
        }
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// Whether a confluent reduction machine is available.
    pub fn is_confluent(&self) -> bool {
        self.machine.as_ref().map(|m| m.is_confluent()).unwrap_or(true)
    }

    /// Create a generator
    pub fn gen(&self, index: usize) -> Option<Element> {
        if index < self.num_generators {
            Some(Element::new(vec![index]))
        } else {
            None
        }
    }

    /// Reduce a word to its normal form using the reduction machine.
    pub fn reduce(&self, word: &[usize]) -> Element {
        let nf = match &self.machine {
            Some(m) => m.reduce(word),
            None => word.to_vec(),
        };
        Element::new(nf)
    }

    /// Multiply two elements: concatenate the normal forms and reduce via the
    /// reduction machine (the real automatic-semigroup product, replacing the
    /// former empty-word stub).
    pub fn mul(&self, a: &Element, b: &Element) -> Element {
        let mut word = a.normal_form.clone();
        word.extend_from_slice(&b.normal_form);
        self.reduce(&word)
    }
}

/// An automatic monoid
#[derive(Debug, Clone)]
pub struct AutomaticMonoid {
    /// The underlying semigroup
    semigroup: AutomaticSemigroup,
}

impl AutomaticMonoid {
    /// Create a new (free) automatic monoid
    pub fn new(num_generators: usize) -> Self {
        AutomaticMonoid {
            semigroup: AutomaticSemigroup::new(num_generators),
        }
    }

    /// Create an automatic monoid from a rewrite presentation.
    pub fn from_relations(
        num_generators: usize,
        relations: &[(Vec<usize>, Vec<usize>)],
        ordering: WordOrdering,
    ) -> Self {
        AutomaticMonoid {
            semigroup: AutomaticSemigroup::from_relations(num_generators, relations, ordering),
        }
    }

    /// Get the identity element (empty word)
    pub fn identity(&self) -> Element {
        Element::new(vec![])
    }

    /// Get the number of generators
    pub fn num_generators(&self) -> usize {
        self.semigroup.num_generators()
    }

    /// Create a generator
    pub fn gen(&self, index: usize) -> Option<Element> {
        self.semigroup.gen(index)
    }

    /// Multiply two elements via the reduction machine.
    pub fn mul(&self, a: &Element, b: &Element) -> Element {
        self.semigroup.mul(a, b)
    }

    /// Reduce a word to normal form.
    pub fn reduce(&self, word: &[usize]) -> Element {
        self.semigroup.reduce(word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_automatic_semigroup() {
        let S = AutomaticSemigroup::new(3);
        assert_eq!(S.num_generators(), 3);
    }

    #[test]
    fn test_automatic_monoid() {
        let M = AutomaticMonoid::new(2);
        assert_eq!(M.num_generators(), 2);

        let id = M.identity();
        assert!(id.normal_form().is_empty());
    }

    #[test]
    fn test_element() {
        let e = Element::new(vec![0, 1, 2]);
        assert_eq!(e.normal_form(), &[0, 1, 2]);
    }

    #[test]
    fn test_gen() {
        let M = AutomaticMonoid::new(5);
        let g2 = M.gen(2).unwrap();
        assert_eq!(g2.normal_form(), &[2]);

        assert!(M.gen(10).is_none());
    }

    #[test]
    fn test_free_mul_is_concatenation() {
        // A free automatic semigroup (no relations): product is concatenation,
        // NOT the empty word (the former stub bug).
        let S = AutomaticSemigroup::new(3);
        let a = S.gen(0).unwrap();
        let b = S.gen(1).unwrap();
        let ab = S.mul(&a, &b);
        assert_eq!(ab.normal_form(), &[0, 1]);
    }

    #[test]
    fn test_mul_reduces_to_normal_form() {
        // ⟨a,b | a^2 = a⟩ : aa reduces to a, so a*a = a and (a*a)*b = a*b.
        let S = AutomaticSemigroup::from_relations(
            2,
            &[(vec![0, 0], vec![0])],
            WordOrdering::ShortLex,
        );
        assert!(S.is_confluent());
        let a = S.gen(0).unwrap();
        let b = S.gen(1).unwrap();
        let aa = S.mul(&a, &a);
        assert_eq!(aa.normal_form(), &[0]); // reduced, not empty
        let aab = S.mul(&aa, &b);
        assert_eq!(aab.normal_form(), &[0, 1]);
    }

    #[test]
    fn test_monoid_mul_commutative_reduction() {
        // ⟨a,b | ba = ab⟩ : b*a reduces to a*b.
        let M = AutomaticMonoid::from_relations(
            2,
            &[(vec![1, 0], vec![0, 1])],
            WordOrdering::ShortLex,
        );
        let a = M.gen(0).unwrap();
        let b = M.gen(1).unwrap();
        assert_eq!(M.mul(&b, &a).normal_form(), &[0, 1]);
        // identity behaves
        let id = M.identity();
        assert_eq!(M.mul(&a, &id).normal_form(), &[0]);
    }
}
