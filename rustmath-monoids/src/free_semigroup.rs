//! Free semigroups and their words (MAGMA handbook chapter 77, §77.2–77.3, 77.8).
//!
//! A free semigroup on `n` generators is the set of *non-empty* finite words
//! over those generators under concatenation. Unlike a free monoid it has no
//! identity element, so its words implement the `rustmath_core`
//! `Magma → Semigroup` tower but *not* `Monoid`.
//!
//! Words are ordered first by length and then lexicographically, with the
//! generator order `S.1 < S.2 < …` (MAGMA §77.3.3). Equality is syntactic
//! (identity as elements of the free semigroup), not modulo any relations.
//!
//! Generator indices are 0-based here (matching the rest of the crate), whereas
//! MAGMA numbers generators from 1.

use rustmath_core::{Magma, Semigroup};
use std::cmp::Ordering;
use std::fmt::{self, Display};

/// A non-empty word in a free semigroup (`GrpFPSgpElt` analogue).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FreeSemigroupElement {
    word: Vec<usize>,
}

impl FreeSemigroupElement {
    /// Create a word from generator indices.
    pub fn new(word: Vec<usize>) -> Self {
        FreeSemigroupElement { word }
    }

    /// The underlying word (0-based generator indices).
    pub fn word(&self) -> &[usize] {
        &self.word
    }

    /// The length of the word (`#u`).
    pub fn len(&self) -> usize {
        self.word.len()
    }

    /// Whether the word is empty. (A genuine semigroup element is never empty;
    /// the empty word can arise transiently from monoid string operations.)
    pub fn is_empty(&self) -> bool {
        self.word.is_empty()
    }

    /// The `i`-th letter (0-based).
    pub fn get(&self, i: usize) -> Option<usize> {
        self.word.get(i).copied()
    }

    /// Concatenation `u * v` (§77.3.1).
    pub fn mul(&self, other: &Self) -> Self {
        let mut w = self.word.clone();
        w.extend_from_slice(&other.word);
        FreeSemigroupElement::new(w)
    }

    /// The `n`-th power `u ^ n` for `n ≥ 1` (§77.3.1).
    pub fn pow(&self, n: usize) -> Self {
        let mut w = Vec::with_capacity(self.word.len() * n);
        for _ in 0..n {
            w.extend_from_slice(&self.word);
        }
        FreeSemigroupElement::new(w)
    }

    /// `ElementToSequence` / `Eltseq` (§77.8): the constituent generator
    /// indices (0-based).
    pub fn eltseq(&self) -> Vec<usize> {
        self.word.clone()
    }

    /// `Subword(u, f, n)` (§77.8): the `n` consecutive letters starting at the
    /// `f`-th (1-based) letter of `u`.
    pub fn subword(&self, f: usize, n: usize) -> FreeSemigroupElement {
        let start = f.saturating_sub(1);
        let end = (start + n).min(self.word.len());
        FreeSemigroupElement::new(self.word[start.min(self.word.len())..end].to_vec())
    }

    /// `Substitute(u, f, n, v)` (§77.8): replace the length-`n` substring of `u`
    /// starting at position `f` (1-based) by the word `v`.
    pub fn substitute(&self, f: usize, n: usize, v: &FreeSemigroupElement) -> FreeSemigroupElement {
        let start = (f.saturating_sub(1)).min(self.word.len());
        let end = (start + n).min(self.word.len());
        let mut w = self.word[..start].to_vec();
        w.extend_from_slice(&v.word);
        w.extend_from_slice(&self.word[end..]);
        FreeSemigroupElement::new(w)
    }

    /// `Eliminate(u, x, v)` (§77.8): replace every occurrence of generator `x`
    /// in `u` by the word `v`.
    pub fn eliminate(&self, x: usize, v: &FreeSemigroupElement) -> FreeSemigroupElement {
        let mut w = Vec::new();
        for &g in &self.word {
            if g == x {
                w.extend_from_slice(&v.word);
            } else {
                w.push(g);
            }
        }
        FreeSemigroupElement::new(w)
    }

    /// `Match(u, v, f)` (§77.8): the least position `l ≥ f` (1-based) at which
    /// `v` occurs as a subword of `u`, or `None`.
    pub fn match_subword(&self, v: &FreeSemigroupElement, f: usize) -> Option<usize> {
        if v.word.is_empty() || v.word.len() > self.word.len() {
            return None;
        }
        let start = f.saturating_sub(1);
        (start..=self.word.len() - v.word.len())
            .find(|&p| self.word[p..p + v.word.len()] == v.word[..])
            .map(|p| p + 1)
    }

    /// `RotateWord(u, n)` (§77.8): cyclically permute `u` by `n` places.
    /// Positive `n` rotates right (left-to-right), negative rotates left,
    /// `n = 0` is the identity.
    pub fn rotate(&self, n: i64) -> FreeSemigroupElement {
        let len = self.word.len();
        if len == 0 {
            return self.clone();
        }
        let shift = n.rem_euclid(len as i64) as usize;
        let mut w = self.word.clone();
        w.rotate_right(shift);
        FreeSemigroupElement::new(w)
    }
}

impl PartialOrd for FreeSemigroupElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FreeSemigroupElement {
    /// Length-then-lexicographic order (§77.3.3).
    fn cmp(&self, other: &Self) -> Ordering {
        match self.word.len().cmp(&other.word.len()) {
            Ordering::Equal => self.word.cmp(&other.word),
            ord => ord,
        }
    }
}

impl Magma for FreeSemigroupElement {
    fn op(&self, other: &Self) -> Self {
        self.mul(other)
    }
}

impl Semigroup for FreeSemigroupElement {}

/// A free semigroup on a fixed number of generators (`FreeSemigroup(n)`).
#[derive(Debug, Clone)]
pub struct FreeSemigroup {
    num_generators: usize,
    generator_names: Vec<String>,
}

impl FreeSemigroup {
    /// `FreeSemigroup(n)`: the free semigroup on `n` generators.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "a free semigroup needs at least one generator");
        let generator_names = (0..n).map(|i| format!("x_{}", i + 1)).collect();
        FreeSemigroup {
            num_generators: n,
            generator_names,
        }
    }

    /// The free semigroup with named generators.
    pub fn with_names(names: Vec<String>) -> Self {
        assert!(!names.is_empty(), "a free semigroup needs at least one generator");
        FreeSemigroup {
            num_generators: names.len(),
            generator_names: names,
        }
    }

    /// `NumberOfGenerators` / `Ngens`.
    pub fn ngens(&self) -> usize {
        self.num_generators
    }

    /// The generator names.
    pub fn generator_names(&self) -> &[String] {
        &self.generator_names
    }

    /// The `i`-th defining generator `S.i` (0-based).
    pub fn gen(&self, i: usize) -> Option<FreeSemigroupElement> {
        if i < self.num_generators {
            Some(FreeSemigroupElement::new(vec![i]))
        } else {
            None
        }
    }

    /// A sequence of the defining generators (`Generators(S)`).
    pub fn generators(&self) -> Vec<FreeSemigroupElement> {
        (0..self.num_generators)
            .map(|i| FreeSemigroupElement::new(vec![i]))
            .collect()
    }

    /// `S ! [i1, …, is]`: the word `S.i1 * … * S.is` (0-based indices).
    pub fn element(&self, indices: &[usize]) -> FreeSemigroupElement {
        FreeSemigroupElement::new(indices.to_vec())
    }

    /// `Random(S, m, n)` (§77.8): a random word of length `l ∈ [m, n]`.
    /// `seed` is advanced so callers get a reproducible stream.
    pub fn random(&self, m: usize, n: usize, seed: &mut u64) -> FreeSemigroupElement {
        let span = n.saturating_sub(m) + 1;
        let len = m + (next_rand(seed) as usize) % span;
        let word = (0..len)
            .map(|_| (next_rand(seed) as usize) % self.num_generators)
            .collect();
        FreeSemigroupElement::new(word)
    }
}

impl Display for FreeSemigroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Free semigroup on {} generators [{}]",
            self.num_generators,
            self.generator_names.join(", ")
        )
    }
}

/// A small splitmix64 step for reproducible `Random`.
fn next_rand(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction_and_generators() {
        let s = FreeSemigroup::new(2);
        assert_eq!(s.ngens(), 2);
        let x = s.gen(0).unwrap();
        let y = s.gen(1).unwrap();
        assert_eq!(x.word(), &[0]);
        assert_eq!(y.word(), &[1]);
        assert!(s.gen(2).is_none());
    }

    #[test]
    fn word_arithmetic_and_length() {
        let s = FreeSemigroup::new(2);
        let x = s.gen(0).unwrap();
        let y = s.gen(1).unwrap();
        let w = x.op(&y).op(&x); // x y x
        assert_eq!(w.word(), &[0, 1, 0]);
        assert_eq!(w.len(), 3);
        assert_eq!(x.pow(3).word(), &[0, 0, 0]);
        // associativity
        assert_eq!(x.op(&y).op(&x), x.op(&y.op(&x)));
    }

    #[test]
    fn length_then_lex_comparison() {
        let s = FreeSemigroup::new(3);
        let a = s.element(&[0]);
        let aa = s.element(&[0, 0]);
        let b = s.element(&[1]);
        let ab = s.element(&[0, 1]);
        let ba = s.element(&[1, 0]);
        assert!(a < aa); // shorter first
        assert!(a < b); // same length, gen order
        assert!(ab < ba); // lex within equal length
        assert!(aa < ab); // 00 < 01
        assert_eq!(ab.cmp(&ab), Ordering::Equal);
    }

    #[test]
    fn string_operations() {
        let s = FreeSemigroup::new(3);
        let u = s.element(&[0, 1, 2, 0, 1]); // x1 x2 x3 x1 x2 (1-based names)
        // Subword: 2 letters from position 2 -> [1,2]
        assert_eq!(u.subword(2, 2).word(), &[1, 2]);
        // Substitute: replace 1 letter at position 3 by [2,2]
        assert_eq!(
            u.substitute(3, 1, &s.element(&[2, 2])).word(),
            &[0, 1, 2, 2, 0, 1]
        );
        // Eliminate generator 0 by word [2]
        assert_eq!(u.eliminate(0, &s.element(&[2])).word(), &[2, 1, 2, 2, 1]);
        // Match subword [0,1] starting from position 1 -> found at 1
        assert_eq!(u.match_subword(&s.element(&[0, 1]), 1), Some(1));
        // Match again after position 2 -> found at 4
        assert_eq!(u.match_subword(&s.element(&[0, 1]), 2), Some(4));
        // No match
        assert_eq!(u.match_subword(&s.element(&[2, 2]), 1), None);
        // Rotate right by 1
        assert_eq!(u.rotate(1).word(), &[1, 0, 1, 2, 0]);
        assert_eq!(u.rotate(0), u);
        // eltseq
        assert_eq!(u.eltseq(), vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn random_is_reproducible_and_in_range() {
        let s = FreeSemigroup::new(4);
        let mut seed = 12345u64;
        for _ in 0..20 {
            let w = s.random(2, 5, &mut seed);
            assert!(w.len() >= 2 && w.len() <= 5);
            assert!(w.word().iter().all(|&g| g < 4));
        }
        // same seed => same first word
        let mut a = 42u64;
        let mut b = 42u64;
        assert_eq!(s.random(1, 3, &mut a), s.random(1, 3, &mut b));
    }
}
