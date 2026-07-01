//! Knuth–Bendix completion for string rewriting systems.
//!
//! Ported from the MAGMA handbook chapters 74 / 75 / 78, which wrap Derek
//! Holt's KBMAG (`[Hol97]`). Given a finite monoid presentation `⟨ X | R ⟩` and
//! a reduction ordering (see [`WordOrdering`]) the completion procedure attempts
//! to produce a finite *confluent* set of rewrite rules — a complete rewriting
//! system — under which every word reduces to a unique irreducible normal form,
//! solving the word problem.
//!
//! Completion need not terminate, so [`RewritingSystem::complete`] takes a
//! [`KbLimits`] budget and honestly reports whether the result is confluent
//! (`true`) or was cut off (`false`), mirroring MAGMA's confluent /
//! non-confluent distinction.
//!
//! This engine is the shared machinery consumed by rewrite monoids (ch78,
//! `rustmath-monoids`) and rewrite / automatic groups (ch74/75).

use crate::word_ordering::WordOrdering;

/// A word over the generator alphabet `{0, …, n-1}`.
pub type Word = Vec<usize>;

/// A rewrite rule `lhs → rhs` with `lhs > rhs` under the ambient ordering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rule {
    /// Left-hand side (the reducible pattern). Never empty.
    pub lhs: Word,
    /// Right-hand side (the replacement).
    pub rhs: Word,
}

/// Resource limits bounding the (possibly non-terminating) completion loop.
///
/// Field names follow the MAGMA `RWSMonoid` parameters where applicable.
#[derive(Debug, Clone)]
pub struct KbLimits {
    /// Maximum number of reduction equations kept (`MaxRelations`, MAGMA 32767).
    pub max_relations: usize,
    /// Maximum number of completion passes before giving up.
    pub max_passes: usize,
}

impl Default for KbLimits {
    fn default() -> Self {
        KbLimits {
            max_relations: 32767,
            max_passes: 200,
        }
    }
}

/// A string rewriting system: an alphabet size, an ordering and a rule set.
#[derive(Debug, Clone)]
pub struct RewritingSystem {
    num_generators: usize,
    ordering: WordOrdering,
    rules: Vec<Rule>,
    confluent: bool,
}

impl RewritingSystem {
    /// An empty system on `num_generators` letters with the given ordering.
    pub fn new(num_generators: usize, ordering: WordOrdering) -> Self {
        RewritingSystem {
            num_generators,
            ordering,
            rules: Vec::new(),
            confluent: false,
        }
    }

    /// Build a system from a list of relations `l = r`, orienting each into a
    /// rule. The result is *not* completed.
    pub fn from_relations(
        num_generators: usize,
        ordering: WordOrdering,
        relations: &[(Word, Word)],
    ) -> Self {
        let mut sys = RewritingSystem::new(num_generators, ordering);
        for (l, r) in relations {
            sys.add_relation(l, r);
        }
        sys
    }

    /// Number of generators (alphabet size).
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// The ambient word ordering.
    pub fn ordering(&self) -> &WordOrdering {
        &self.ordering
    }

    /// The current rule set (a complete rewriting system after successful
    /// completion). Each rule's LHS exceeds its RHS under the ordering.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// `true` iff the last [`complete`](Self::complete) call proved confluence.
    pub fn is_confluent(&self) -> bool {
        self.confluent
    }

    /// Add a single relation `l = r`, orienting it. Trivial relations
    /// (`l == r`, or reducing to it) are ignored.
    pub fn add_relation(&mut self, l: &[usize], r: &[usize]) {
        if let Some((lhs, rhs)) = self.ordering.orient(l, r) {
            if lhs != rhs && !lhs.is_empty() {
                self.rules.push(Rule { lhs, rhs });
            }
        }
        self.confluent = false;
    }

    /// Reduce `w` to normal form with respect to the current rules by
    /// repeatedly rewriting the first applicable factor until irreducible.
    pub fn reduce(&self, w: &[usize]) -> Word {
        reduce_with(&self.rules, w)
    }

    /// Run Knuth–Bendix completion within the given limits.
    ///
    /// Returns `true` if a finite confluent system was found, `false` if a
    /// limit was hit first (the partial, non-confluent system is retained).
    pub fn complete(&mut self, limits: &KbLimits) -> bool {
        self.interreduce();
        for _ in 0..limits.max_passes {
            let new_rules = self.collect_critical_pairs();
            let mut added = false;
            for (lhs, rhs) in new_rules {
                if self.rules.len() >= limits.max_relations {
                    self.confluent = false;
                    return false;
                }
                if !self.rules.iter().any(|r| r.lhs == lhs && r.rhs == rhs) {
                    self.rules.push(Rule { lhs, rhs });
                    added = true;
                }
            }
            self.interreduce();
            if !added {
                self.confluent = true;
                return true;
            }
        }
        self.confluent = false;
        false
    }

    /// Independent confluence test: check that every critical pair resolves to
    /// a common normal form under the current rules.
    pub fn check_confluent(&self) -> bool {
        self.collect_critical_pairs().is_empty()
    }

    /// Gather the unresolved critical pairs of the current rule set, each
    /// oriented into a would-be rule `(lhs, rhs)`.
    fn collect_critical_pairs(&self) -> Vec<(Word, Word)> {
        let mut out: Vec<(Word, Word)> = Vec::new();
        let n = self.rules.len();
        for i in 0..n {
            for j in 0..n {
                self.critical_pairs_between(i, j, &mut out);
            }
        }
        out
    }

    /// Critical pairs arising from rule `i` overlapping rule `j`.
    fn critical_pairs_between(&self, i: usize, j: usize, out: &mut Vec<(Word, Word)>) {
        let (l1, r1) = (&self.rules[i].lhs, &self.rules[i].rhs);
        let (l2, r2) = (&self.rules[j].lhs, &self.rules[j].rhs);

        // Suffix/prefix overlaps: a nonempty proper suffix of l1 equals a
        // nonempty proper prefix of l2. Overlap word = l1 · l2[k..].
        let max_k = l1.len().min(l2.len());
        for k in 1..max_k {
            if l1[l1.len() - k..] == l2[..k] {
                // critical word: l1 followed by the tail of l2
                let mut a = r1.clone(); // rewrite the l1 occurrence
                a.extend_from_slice(&l2[k..]);
                let mut b = l1[..l1.len() - k].to_vec(); // rewrite the l2 occurrence
                b.extend_from_slice(r2);
                self.push_pair(&a, &b, out);
            }
        }

        // Containment overlaps: l2 occurs as a factor strictly inside l1.
        if i != j && l2.len() <= l1.len() {
            let mut p = 0;
            while p + l2.len() <= l1.len() {
                if l1[p..p + l2.len()] == l2[..] {
                    let mut b = l1[..p].to_vec();
                    b.extend_from_slice(r2);
                    b.extend_from_slice(&l1[p + l2.len()..]);
                    // one side rewrites via rule i, the other via rule j
                    self.push_pair(r1, &b, out);
                }
                p += 1;
            }
        }
    }

    fn push_pair(&self, a: &[usize], b: &[usize], out: &mut Vec<(Word, Word)>) {
        let na = self.reduce(a);
        let nb = self.reduce(b);
        if na != nb {
            if let Some((lhs, rhs)) = self.ordering.orient(&na, &nb) {
                if !lhs.is_empty() {
                    out.push((lhs, rhs));
                }
            }
        }
    }

    /// Tidy the rule set (mirroring KBMAG's `TidyInt` interreduction step).
    ///
    /// This is Huet's collapse/simplify step performed *one rule at a time*
    /// against the live rule set: pick any rule whose LHS or RHS is reducible by
    /// the other rules, remove it, and re-add the re-oriented reduced equation
    /// unless it has become trivial (in which case the rule was redundant).
    /// Working sequentially — rather than reducing all rules against a single
    /// snapshot — is essential: it prevents a set of mutually-redundant rules
    /// from all collapsing at once and silently dropping a relation. A
    /// well-founded measure (the multiset of LHSs under the reduction ordering)
    /// strictly decreases on each change, so this terminates.
    fn interreduce(&mut self) {
        loop {
            let mut acted = false;
            for i in 0..self.rules.len() {
                let others: Vec<Rule> = self
                    .rules
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != i)
                    .map(|(_, r)| r.clone())
                    .collect();
                let l_new = reduce_with(&others, &self.rules[i].lhs);
                let r_new = reduce_with(&others, &self.rules[i].rhs);
                if l_new == self.rules[i].lhs && r_new == self.rules[i].rhs {
                    continue; // this rule is already fully reduced
                }
                // Rule i can be simplified: drop it and re-add the reduced,
                // re-oriented equation (if non-trivial).
                self.rules.remove(i);
                if let Some((lhs, rhs)) = self.ordering.orient(&l_new, &r_new) {
                    if lhs != rhs
                        && !lhs.is_empty()
                        && !self.rules.iter().any(|d| d.lhs == lhs && d.rhs == rhs)
                    {
                        self.rules.push(Rule { lhs, rhs });
                    }
                }
                acted = true;
                break; // restart the scan against the updated set
            }
            if !acted {
                break;
            }
        }
        // Present rules in ordering-sorted order (LHS ascending) for stable,
        // MAGMA-like `Relations` output.
        let ord = self.ordering.clone();
        self.rules.sort_by(|a, b| ord.compare(&a.lhs, &b.lhs));
    }
}

/// Find the first (leftmost) position at which `pat` occurs as a factor of `w`.
fn find_factor(w: &[usize], pat: &[usize]) -> Option<usize> {
    if pat.is_empty() || pat.len() > w.len() {
        return None;
    }
    (0..=w.len() - pat.len()).find(|&p| w[p..p + pat.len()] == pat[..])
}

/// Reduce `w` to normal form using an explicit slice of rules.
pub fn reduce_with(rules: &[Rule], w: &[usize]) -> Word {
    let mut word = w.to_vec();
    'outer: loop {
        for rule in rules {
            if let Some(pos) = find_factor(&word, &rule.lhs) {
                let mut nw = Vec::with_capacity(word.len());
                nw.extend_from_slice(&word[..pos]);
                nw.extend_from_slice(&rule.rhs);
                nw.extend_from_slice(&word[pos + rule.lhs.len()..]);
                word = nw;
                continue 'outer;
            }
        }
        return word;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_against_simple_rule() {
        // aa -> a  (idempotent generator)
        let sys = RewritingSystem::from_relations(
            1,
            WordOrdering::ShortLex,
            &[(vec![0, 0], vec![0])],
        );
        assert_eq!(sys.reduce(&[0, 0, 0, 0]), vec![0]);
        assert_eq!(sys.reduce(&[0]), vec![0]);
        assert_eq!(sys.reduce(&[]), Vec::<usize>::new());
    }

    #[test]
    fn completion_of_free_commutative_monoid_on_two_gens() {
        // ⟨ a, b | ba = ab ⟩ : commuting generators.  Under ShortLex the rule
        // ba -> ab is already confluent (no overlaps), normal forms are a^i b^j.
        let mut sys = RewritingSystem::from_relations(
            2,
            WordOrdering::ShortLex,
            &[(vec![1, 0], vec![0, 1])],
        );
        assert!(sys.complete(&KbLimits::default()));
        assert!(sys.is_confluent());
        // b a b a  ->  a a b b
        assert_eq!(sys.reduce(&[1, 0, 1, 0]), vec![0, 0, 1, 1]);
    }

    #[test]
    fn completion_needs_new_rules_klein_four() {
        // Monoid presentation of the Klein four-group C2 x C2:
        // a^2 = 1, b^2 = 1, abab = 1  (=> ba = ab after completion).
        let mut sys = RewritingSystem::from_relations(
            2,
            WordOrdering::ShortLex,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![0, 1, 0, 1], vec![]),
            ],
        );
        assert!(sys.complete(&KbLimits::default()));
        assert!(sys.is_confluent());
        assert!(sys.check_confluent());
        // The four normal forms are 1, a, b, ab. ba reduces to ab.
        assert_eq!(sys.reduce(&[1, 0]), vec![0, 1]);
        assert_eq!(sys.reduce(&[0, 1, 0, 1]), Vec::<usize>::new());
        assert_eq!(sys.reduce(&[1, 1, 0, 0]), Vec::<usize>::new());
    }

    #[test]
    fn non_confluent_is_reported_honestly() {
        // A tiny budget on a presentation needing completion cannot confluence.
        let mut sys = RewritingSystem::from_relations(
            2,
            WordOrdering::ShortLex,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![0, 1, 0, 1], vec![]),
            ],
        );
        let tight = KbLimits {
            max_relations: 3,
            max_passes: 1,
        };
        assert!(!sys.complete(&tight));
        assert!(!sys.is_confluent());
    }
}
