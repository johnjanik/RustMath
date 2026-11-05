//! Finite-state reduction machine and word-acceptor for a confluent rewriting
//! system.
//!
//! Ported from MAGMA handbook chapters 74 / 75 / 78 (`[Hol97]`). After
//! Knuth–Bendix completion produces a confluent rule set, this module compiles
//! it into two finite automata:
//!
//! * an **index automaton** (Aho–Corasick machine) over the generator alphabet
//!   that recognises rule left-hand sides, used to reduce words to normal form
//!   (the *reduction machine*); and
//! * a **word acceptor** DFA — reusing [`crate::DFA`] — that accepts exactly the
//!   irreducible (normal-form) words. This is the automaton MAGMA exposes via
//!   `WordAcceptor`/`WordAcceptorSize`, and from which `Order`, `IsFinite`,
//!   `GrowthFunction` and element enumeration (`Set`/`Seq`) are derived.

use crate::knuth_bendix::{reduce_with, RewritingSystem, Rule, Word};
use crate::word_ordering::WordOrdering;
use crate::DFA;
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;
use std::collections::{HashMap, HashSet, VecDeque};

/// A rational generating function `numerator / denominator` with integer
/// coefficients (low-degree first), as returned by MAGMA's `GrowthFunction`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrowthFunction {
    /// Numerator polynomial, coefficients from constant term upward.
    pub numerator: Vec<Integer>,
    /// Denominator polynomial, coefficients from constant term upward.
    pub denominator: Vec<Integer>,
}

impl GrowthFunction {
    /// The numerator as a [`UnivariatePolynomial`] over the integers.
    pub fn numerator_polynomial(&self) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(self.numerator.clone())
    }

    /// The denominator as a [`UnivariatePolynomial`] over the integers.
    pub fn denominator_polynomial(&self) -> UnivariatePolynomial<Integer> {
        UnivariatePolynomial::new(self.denominator.clone())
    }
}

/// A compiled reduction machine for a (confluent) rewriting system.
///
/// This is the shared object consumed by rewrite monoids (ch78) and rewrite /
/// automatic groups (ch74/75). It is deliberately concrete (no trait objects),
/// as the port plan requires.
#[derive(Debug, Clone)]
pub struct ReductionAutomaton {
    num_generators: usize,
    ordering: WordOrdering,
    rules: Vec<Rule>,
    confluent: bool,
    /// Optional involution on letters (`letter -> inverse letter`), present for
    /// group presentations over generators-and-inverses.
    inverse_map: Option<Vec<usize>>,
    /// Word-acceptor transition graph: `adj[state][letter] = Some(next state)`
    /// when reading `letter` keeps the word irreducible, else `None`. State `0`
    /// is the start; every state is accepting.
    adj: Vec<Vec<Option<usize>>>,
}

impl ReductionAutomaton {
    /// Compile a completed rewriting system into a reduction machine.
    ///
    /// `inverse_map`, if given, records the letter involution for group
    /// presentations (used by [`crate::rws_group`] and friends).
    pub fn from_system(sys: &RewritingSystem, inverse_map: Option<Vec<usize>>) -> Self {
        let adj = build_acceptor(sys.num_generators(), sys.rules());
        ReductionAutomaton {
            num_generators: sys.num_generators(),
            ordering: sys.ordering().clone(),
            rules: sys.rules().to_vec(),
            confluent: sys.is_confluent(),
            inverse_map,
            adj,
        }
    }

    /// Alphabet size.
    pub fn num_generators(&self) -> usize {
        self.num_generators
    }

    /// The confluent rewrite rules.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// The word ordering used.
    pub fn ordering(&self) -> &WordOrdering {
        &self.ordering
    }

    /// Whether the underlying system was proved confluent.
    pub fn is_confluent(&self) -> bool {
        self.confluent
    }

    /// The letter involution, for group presentations.
    pub fn inverse_map(&self) -> Option<&[usize]> {
        self.inverse_map.as_deref()
    }

    /// Reduce a word to its normal form via the rewrite rules.
    pub fn reduce(&self, w: &[usize]) -> Word {
        reduce_with(&self.rules, w)
    }

    /// Number of states of the word acceptor.
    pub fn word_acceptor_size(&self) -> usize {
        self.adj.len()
    }

    /// The word-acceptor as a reusable [`DFA`]; accepts exactly the normal-form
    /// words. Alphabet is `{0, …, n-1}`, states are `0, …, size-1`, all
    /// accepting, start state `0`.
    pub fn word_acceptor_dfa(&self) -> DFA<usize, usize> {
        let states: HashSet<usize> = (0..self.adj.len()).collect();
        let alphabet: HashSet<usize> = (0..self.num_generators).collect();
        let mut transitions: HashMap<(usize, usize), usize> = HashMap::new();
        for (s, row) in self.adj.iter().enumerate() {
            for (c, t) in row.iter().enumerate() {
                if let Some(t) = t {
                    transitions.insert((s, c), *t);
                }
            }
        }
        let accepting = states.clone();
        DFA::new(states, alphabet, transitions, 0, accepting)
            .expect("well-formed word-acceptor DFA")
    }

    /// The order of the monoid/group, i.e. the number of normal-form words.
    /// Returns `None` when infinite (MAGMA's `∞`).
    pub fn order(&self) -> Option<Integer> {
        if self.has_cycle() {
            return None;
        }
        // Acyclic: number of words = number of paths from the start; every
        // state accepts, so ways(s) = 1 + sum over out-edges ways(t).
        let mut memo: Vec<Option<Integer>> = vec![None; self.adj.len()];
        Some(self.count_ways(0, &mut memo))
    }

    /// Whether the order is finite (with the order if so). Mirrors
    /// MAGMA's `IsFinite`.
    pub fn is_finite(&self) -> (bool, Option<Integer>) {
        match self.order() {
            Some(n) => (true, Some(n)),
            None => (false, None),
        }
    }

    fn count_ways(&self, s: usize, memo: &mut Vec<Option<Integer>>) -> Integer {
        if let Some(v) = &memo[s] {
            return v.clone();
        }
        let mut total = Integer::one(); // the empty continuation (state accepts)
        for c in 0..self.num_generators {
            if let Some(t) = self.adj[s][c] {
                total = total + self.count_ways(t, memo);
            }
        }
        memo[s] = Some(total.clone());
        total
    }

    /// Detect a directed cycle reachable from the start state.
    fn has_cycle(&self) -> bool {
        let n = self.adj.len();
        // 0 = white, 1 = grey (on stack), 2 = black
        let mut color = vec![0u8; n];
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)]; // (state, next letter)
        color[0] = 1;
        while let Some(&(s, ci)) = stack.last() {
            if ci >= self.num_generators {
                color[s] = 2;
                stack.pop();
                continue;
            }
            stack.last_mut().unwrap().1 += 1;
            if let Some(t) = self.adj[s][ci] {
                match color[t] {
                    1 => return true, // back edge => cycle
                    0 => {
                        color[t] = 1;
                        stack.push((t, 0));
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// The growth series: `coeffs[k]` = number of normal-form words of length
    /// `k`, for `k = 0, …, nmax`.
    pub fn growth_series(&self, nmax: usize) -> Vec<Integer> {
        let n = self.adj.len();
        let mut level = vec![Integer::zero(); n];
        level[0] = Integer::one();
        let mut coeffs = Vec::with_capacity(nmax + 1);
        for _ in 0..=nmax {
            // number of words of the current length = sum over states (all accept)
            let mut c = Integer::zero();
            for v in &level {
                c = c + v.clone();
            }
            coeffs.push(c);
            // advance one letter
            let mut next = vec![Integer::zero(); n];
            for (s, cnt) in level.iter().enumerate() {
                if cnt.is_zero() {
                    continue;
                }
                for cc in 0..self.num_generators {
                    if let Some(t) = self.adj[s][cc] {
                        next[t] = next[t].clone() + cnt.clone();
                    }
                }
            }
            level = next;
        }
        coeffs
    }

    /// The rational growth function (MAGMA `GrowthFunction`): a rational
    /// function whose Taylor coefficient of `x^n` is the number of normal-form
    /// words of length `n`. Computed from the word-acceptor via the
    /// Berlekamp–Massey minimal linear recurrence, so the result is in lowest
    /// terms. For a finite group the denominator is `1` (a polynomial).
    pub fn growth_function(&self) -> GrowthFunction {
        let m = self.adj.len();
        let terms = 2 * m + 4;
        let series = self.growth_series(terms);
        let s: Vec<Rational> = series
            .iter()
            .map(|x| Rational::from_integer(x.clone()))
            .collect();
        let c = berlekamp_massey(&s); // connection polynomial, c[0] = 1
        let l = c.len().saturating_sub(1);
        // numerator P = (C * S) truncated to degree < l
        let mut p: Vec<Rational> = vec![Rational::from_integer(Integer::zero()); l];
        for (k, pk) in p.iter_mut().enumerate() {
            let mut acc = Rational::from_integer(Integer::zero());
            for (i, ci) in c.iter().enumerate() {
                if i <= k {
                    acc = acc + ci.clone() * s[k - i].clone();
                }
            }
            *pk = acc;
        }
        let (num, den) = clear_denominators(&p, &c);
        GrowthFunction {
            numerator: trim(num),
            denominator: trim(den),
        }
    }

    /// Enumerate normal-form words of length in `[a, b]`. When `bfs` is true,
    /// words come in short-lex order (length then lexicographic); otherwise in
    /// depth-first lexicographic order. Mirrors MAGMA `Seq(M, a, b : Search)`.
    pub fn enumerate(&self, a: usize, b: usize, bfs: bool) -> Vec<Word> {
        let mut out: Vec<Word> = Vec::new();
        if bfs {
            let mut queue: VecDeque<(usize, Word)> = VecDeque::new();
            queue.push_back((0, Vec::new()));
            while let Some((s, w)) = queue.pop_front() {
                if w.len() >= a && w.len() <= b {
                    out.push(w.clone());
                }
                if w.len() < b {
                    for c in 0..self.num_generators {
                        if let Some(t) = self.adj[s][c] {
                            let mut nw = w.clone();
                            nw.push(c);
                            queue.push_back((t, nw));
                        }
                    }
                }
            }
        } else {
            self.dfs_enumerate(0, &mut Vec::new(), a, b, &mut out);
        }
        out
    }

    fn dfs_enumerate(
        &self,
        s: usize,
        w: &mut Word,
        a: usize,
        b: usize,
        out: &mut Vec<Word>,
    ) {
        if w.len() >= a && w.len() <= b {
            out.push(w.clone());
        }
        if w.len() < b {
            for c in 0..self.num_generators {
                if let Some(t) = self.adj[s][c] {
                    w.push(c);
                    self.dfs_enumerate(t, w, a, b, out);
                    w.pop();
                }
            }
        }
    }
}

/// Build the word-acceptor transition graph from the rule left-hand sides
/// using an Aho–Corasick index automaton. `adj[state][letter]` is `Some(next)`
/// when reading `letter` keeps the scanned word free of any rule LHS factor.
fn build_acceptor(alphabet: usize, rules: &[Rule]) -> Vec<Vec<Option<usize>>> {
    // --- build the Aho–Corasick trie of the LHS patterns ---
    let mut goto: Vec<Vec<Option<usize>>> = vec![vec![None; alphabet]];
    let mut terminal: Vec<bool> = vec![false];
    for rule in rules {
        let mut cur = 0;
        for &ch in &rule.lhs {
            if ch >= alphabet {
                continue; // defensive: ignore out-of-range letters
            }
            match goto[cur][ch] {
                Some(nx) => cur = nx,
                None => {
                    let nx = goto.len();
                    goto.push(vec![None; alphabet]);
                    terminal.push(false);
                    goto[cur][ch] = Some(nx);
                    cur = nx;
                }
            }
        }
        if !rule.lhs.is_empty() {
            terminal[cur] = true;
        }
    }

    let nnodes = goto.len();
    let mut delta = vec![vec![0usize; alphabet]; nnodes];
    let mut fail = vec![0usize; nnodes];
    let mut has_match = terminal.clone();

    // depth-1 nodes
    let mut queue: VecDeque<usize> = VecDeque::new();
    for c in 0..alphabet {
        match goto[0][c] {
            Some(nx) => {
                delta[0][c] = nx;
                fail[nx] = 0;
                queue.push_back(nx);
            }
            None => {
                delta[0][c] = 0;
            }
        }
    }
    while let Some(u) = queue.pop_front() {
        // a state matches if it terminates a pattern or its fail state does
        if has_match[fail[u]] {
            has_match[u] = true;
        }
        for c in 0..alphabet {
            match goto[u][c] {
                Some(v) => {
                    fail[v] = delta[fail[u]][c];
                    delta[u][c] = v;
                    queue.push_back(v);
                }
                None => {
                    delta[u][c] = delta[fail[u]][c];
                }
            }
        }
    }

    // --- restrict to "good" (non-matching) states reachable from the root ---
    // Compact-index only good states; edges into matching states are dropped.
    let mut compact: HashMap<usize, usize> = HashMap::new();
    compact.insert(0, 0);
    let mut order: Vec<usize> = vec![0];
    let mut i = 0;
    while i < order.len() {
        let s = order[i];
        i += 1;
        for c in 0..alphabet {
            let t = delta[s][c];
            if !has_match[t] && !compact.contains_key(&t) {
                let id = order.len();
                compact.insert(t, id);
                order.push(t);
            }
        }
    }
    let mut adj: Vec<Vec<Option<usize>>> = vec![vec![None; alphabet]; order.len()];
    for (&ac_state, &cid) in &compact {
        for c in 0..alphabet {
            let t = delta[ac_state][c];
            if !has_match[t] {
                adj[cid][c] = Some(compact[&t]);
            }
        }
    }
    adj
}

/// Berlekamp–Massey over the rationals: return the minimal connection
/// polynomial `C(x) = 1 + c_1 x + … + c_L x^L` for the sequence `s`.
fn berlekamp_massey(s: &[Rational]) -> Vec<Rational> {
    let zero = Rational::from_integer(Integer::zero());
    let one = Rational::from_integer(Integer::one());
    let mut c = vec![one.clone()];
    let mut b = vec![one.clone()];
    let mut l = 0usize;
    let mut m = 1usize;
    let mut bb = one.clone();
    for n in 0..s.len() {
        // discrepancy
        let mut d = s[n].clone();
        for i in 1..=l {
            d = d + c[i].clone() * s[n - i].clone();
        }
        if d == zero {
            m += 1;
        } else if 2 * l <= n {
            let t = c.clone();
            let coef = d.clone() / bb.clone();
            while c.len() < b.len() + m {
                c.push(zero.clone());
            }
            for (i, bi) in b.iter().enumerate() {
                c[i + m] = c[i + m].clone() - coef.clone() * bi.clone();
            }
            l = n + 1 - l;
            b = t;
            bb = d;
            m = 1;
        } else {
            let coef = d.clone() / bb.clone();
            while c.len() < b.len() + m {
                c.push(zero.clone());
            }
            for (i, bi) in b.iter().enumerate() {
                c[i + m] = c[i + m].clone() - coef.clone() * bi.clone();
            }
            m += 1;
        }
    }
    c.truncate(l + 1);
    c
}

/// Scale numerator `p` and denominator `c` (rational-coefficient polynomials) to
/// coprime integer polynomials, normalising so the denominator's constant term
/// is positive.
fn clear_denominators(p: &[Rational], c: &[Rational]) -> (Vec<Integer>, Vec<Integer>) {
    let mut denom_lcm = Integer::one();
    for r in p.iter().chain(c.iter()) {
        denom_lcm = denom_lcm.lcm(r.denominator());
    }
    let scale = |r: &Rational| -> Integer {
        // r * denom_lcm is an integer
        (r.numerator().clone() * denom_lcm.clone()) / r.denominator().clone()
    };
    let mut num: Vec<Integer> = p.iter().map(scale).collect();
    let mut den: Vec<Integer> = c.iter().map(scale).collect();

    // divide out the common integer content
    let mut g = Integer::zero();
    for v in num.iter().chain(den.iter()) {
        g = g.gcd(v);
    }
    if !g.is_zero() && g != Integer::one() {
        for v in num.iter_mut() {
            *v = v.clone() / g.clone();
        }
        for v in den.iter_mut() {
            *v = v.clone() / g.clone();
        }
    }

    // normalise sign so the denominator's constant term is positive
    let const_term = den.first().cloned().unwrap_or_else(Integer::one);
    if const_term.signum() < 0 {
        for v in num.iter_mut() {
            *v = -v.clone();
        }
        for v in den.iter_mut() {
            *v = -v.clone();
        }
    }
    (num, den)
}

/// Drop trailing zero coefficients (keeping at least one).
fn trim(mut v: Vec<Integer>) -> Vec<Integer> {
    while v.len() > 1 && v.last().map(|x| x.is_zero()).unwrap_or(false) {
        v.pop();
    }
    if v.is_empty() {
        v.push(Integer::zero());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knuth_bendix::KbLimits;

    fn machine(
        n: usize,
        ordering: WordOrdering,
        rels: &[(Word, Word)],
    ) -> ReductionAutomaton {
        let mut sys = RewritingSystem::from_relations(n, ordering, rels);
        assert!(sys.complete(&KbLimits::default()));
        ReductionAutomaton::from_system(&sys, None)
    }

    #[test]
    fn klein_four_order_and_acceptor() {
        // ⟨a,b | a^2, b^2, (ab)^2⟩  ->  order 4
        let m = machine(
            2,
            WordOrdering::ShortLex,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![0, 1, 0, 1], vec![]),
            ],
        );
        assert_eq!(m.order(), Some(Integer::from(4u64)));
        let (fin, ord) = m.is_finite();
        assert!(fin);
        assert_eq!(ord, Some(Integer::from(4u64)));
        // normal forms: "", a, b, ab
        let words = m.enumerate(0, 4, true);
        assert_eq!(words.len(), 4);
        assert!(words.contains(&vec![]));
        assert!(words.contains(&vec![0]));
        assert!(words.contains(&vec![1]));
        assert!(words.contains(&vec![0, 1]));
    }

    #[test]
    fn word_acceptor_dfa_recognises_normal_forms() {
        let m = machine(
            2,
            WordOrdering::ShortLex,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![0, 1, 0, 1], vec![]),
            ],
        );
        let dfa = m.word_acceptor_dfa();
        assert!(dfa.accepts(&[0, 1])); // ab is a normal form
        assert!(!dfa.accepts(&[0, 0])); // aa is reducible
        assert!(!dfa.accepts(&[1, 0])); // ba reduces to ab
    }

    #[test]
    fn infinite_free_monoid_reports_infinite_order() {
        // no relations: the free monoid on 1 generator is infinite (a^n)
        let m = machine(1, WordOrdering::ShortLex, &[]);
        assert_eq!(m.order(), None);
        let series = m.growth_series(4);
        assert_eq!(
            series,
            vec![
                Integer::from(1u64),
                Integer::from(1u64),
                Integer::from(1u64),
                Integer::from(1u64),
                Integer::from(1u64)
            ]
        );
    }

    #[test]
    fn growth_function_finite_is_a_polynomial() {
        // Klein four monoid ⟨a,b | a^2, b^2, (ab)^2⟩.  Normal forms "", a, b, ab
        // have lengths 0,1,1,2, so the growth polynomial is 1 + 2x + x^2 and the
        // denominator is 1 (a finite group has polynomial growth).
        let m = machine(
            2,
            WordOrdering::ShortLex,
            &[
                (vec![0, 0], vec![]),
                (vec![1, 1], vec![]),
                (vec![0, 1, 0, 1], vec![]),
            ],
        );
        assert_eq!(m.order(), Some(Integer::from(4u64)));
        let g = m.growth_function();
        assert_eq!(g.denominator, vec![Integer::from(1u64)]);
        assert_eq!(
            g.numerator,
            vec![
                Integer::from(1u64),
                Integer::from(2u64),
                Integer::from(1u64)
            ]
        );
    }

    #[test]
    fn growth_function_infinite_dihedral() {
        // Infinite dihedral: ⟨a,b | a^2, b^2⟩.  Growth series 1,2,2,2,...
        // Generating function (1+x)/(1-x).
        let m = machine(
            2,
            WordOrdering::ShortLex,
            &[(vec![0, 0], vec![]), (vec![1, 1], vec![])],
        );
        assert_eq!(m.order(), None);
        let series = m.growth_series(4);
        assert_eq!(
            series,
            vec![
                Integer::from(1u64),
                Integer::from(2u64),
                Integer::from(2u64),
                Integer::from(2u64),
                Integer::from(2u64)
            ]
        );
        let g = m.growth_function();
        // (1 + x) / (1 - x)
        assert_eq!(g.numerator, vec![Integer::from(1u64), Integer::from(1u64)]);
        assert_eq!(g.denominator, vec![Integer::from(1u64), Integer::from(-1i64)]);
    }
}
