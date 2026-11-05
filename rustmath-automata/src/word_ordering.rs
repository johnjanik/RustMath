//! Word orderings for Knuth–Bendix completion.
//!
//! Ported from the MAGMA handbook, chapters 74 (Groups Defined by Rewrite
//! Systems), 75 (Automatic Groups) and 78 (Monoids Given by Rewrite Systems).
//! MAGMA supports the orderings `ShortLex`, `Recursive`, `RtRecursive`
//! (`RTRecursive`), `WtLex` / `WTShortLex` (weighted short-lex) and `Wreath`
//! (wreath-product ordering, Sims *Computation with finitely presented groups*,
//! pp. 46–50).
//!
//! Every variant here is a *reduction ordering* on the free monoid over the
//! generator alphabet `{0, 1, …, n-1}`: a total well-order that is translation
//! invariant (`u < v  ⟹  x·u·y < x·v·y`). That property is exactly what the
//! Knuth–Bendix completion procedure requires to orient equations into
//! terminating rewrite rules.
//!
//! The generator order is the numeric index order `0 < 1 < 2 < …`, matching
//! MAGMA's `S.1 < S.2 < S.3 < …`.

use std::cmp::Ordering;

/// A term (word) ordering used to orient rewrite rules.
///
/// Words are slices of generator indices (`&[usize]`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordOrdering {
    /// Short-lex: compare by length, then lexicographically. The MAGMA default.
    ShortLex,
    /// Recursive path ordering, scanning words right-to-left. MAGMA `Recursive`.
    Recursive,
    /// Recursive path ordering, scanning words left-to-right. MAGMA `RtRecursive`.
    RtRecursive,
    /// Weighted short-lex: compare by total weight, then length, then lex.
    /// MAGMA `WtLex` / `WTShortLex`. One non-negative weight per generator.
    WtLex(Vec<u64>),
    /// Wreath-product ordering with one non-negative level per generator.
    /// MAGMA `Wreath` (Sims, pp. 46–50).
    ///
    /// Implemented as the recursive path ordering induced by the `(level,
    /// index)` letter key: letters are compared first by level and then by
    /// index. This coincides with [`WordOrdering::Recursive`] for the canonical
    /// strictly-increasing level assignment `level(i) = i`, and is a valid
    /// reduction ordering for any levels. (Sims' short-lex refinement *within*
    /// an equal-level block is not reproduced — a documented deviation that does
    /// not affect termination or confluence of completion.)
    Wreath(Vec<u64>),
}

impl WordOrdering {
    /// Compare two words under this ordering.
    pub fn compare(&self, u: &[usize], v: &[usize]) -> Ordering {
        match self {
            WordOrdering::ShortLex => shortlex(u, v),
            WordOrdering::WtLex(weights) => wtlex(u, v, weights),
            WordOrdering::Recursive => recursive_cmp(u, v),
            WordOrdering::RtRecursive => rtrecursive_cmp(u, v),
            WordOrdering::Wreath(levels) => wreath_cmp(u, v, levels),
        }
    }

    /// `true` iff `u` is strictly greater than `v`.
    pub fn is_greater(&self, u: &[usize], v: &[usize]) -> bool {
        self.compare(u, v) == Ordering::Greater
    }

    /// Orient a relation `l = r` into a rewrite rule `(lhs, rhs)` with
    /// `lhs > rhs`. Returns `None` when `l` and `r` are equal (trivial).
    pub fn orient(&self, l: &[usize], r: &[usize]) -> Option<(Vec<usize>, Vec<usize>)> {
        match self.compare(l, r) {
            Ordering::Greater => Some((l.to_vec(), r.to_vec())),
            Ordering::Less => Some((r.to_vec(), l.to_vec())),
            Ordering::Equal => None,
        }
    }

    /// A short human-readable name matching MAGMA's `Ordering(G)` output.
    pub fn name(&self) -> &'static str {
        match self {
            WordOrdering::ShortLex => "ShortLex",
            WordOrdering::Recursive => "Recursive",
            WordOrdering::RtRecursive => "RtRecursive",
            WordOrdering::WtLex(_) => "WtLex",
            WordOrdering::Wreath(_) => "Wreath",
        }
    }
}

fn shortlex(u: &[usize], v: &[usize]) -> Ordering {
    match u.len().cmp(&v.len()) {
        Ordering::Equal => u.cmp(v),
        ord => ord,
    }
}

fn weight(w: &[usize], weights: &[u64]) -> u64 {
    w.iter()
        .map(|&g| weights.get(g).copied().unwrap_or(1))
        .sum()
}

fn wtlex(u: &[usize], v: &[usize], weights: &[u64]) -> Ordering {
    match weight(u, weights).cmp(&weight(v, weights)) {
        Ordering::Equal => shortlex(u, v),
        ord => ord,
    }
}

// The recursive orderings are implemented as the recursive path ordering (RPO)
// on words viewed as nested unary terms, with the letter precedence supplied by
// a key function. The RPO is a standard total reduction ordering: for terms
// `s = f(s')`, `t = g(t')` (unary), `s ≻ t` iff
//   (subterm)     s' ⪰ t, or
//   (precedence)  [ f > g  or  (f = g and s' ≻ t') ]  and  s ≻ t'.
// `recursive` reads the outermost symbol from the right, `rtrecursive` from the
// left, and `wreath` uses the `(level, index)` key.

/// RPO with the last letter as the outermost symbol; `key` orders letters.
fn rpo_right_greater<F>(u: &[usize], v: &[usize], key: &F) -> bool
where
    F: Fn(usize) -> (u64, usize),
{
    if v.is_empty() {
        return !u.is_empty();
    }
    if u.is_empty() {
        return false;
    }
    let ua = &u[..u.len() - 1];
    let vb = &v[..v.len() - 1];
    if rpo_right_geq(ua, v, key) {
        return true;
    }
    let a = key(u[u.len() - 1]);
    let b = key(v[v.len() - 1]);
    let prec = a > b || (a == b && rpo_right_greater(ua, vb, key));
    prec && rpo_right_greater(u, vb, key)
}

fn rpo_right_geq<F>(u: &[usize], v: &[usize], key: &F) -> bool
where
    F: Fn(usize) -> (u64, usize),
{
    u == v || rpo_right_greater(u, v, key)
}

fn rpo_right_cmp<F>(u: &[usize], v: &[usize], key: &F) -> Ordering
where
    F: Fn(usize) -> (u64, usize),
{
    if u == v {
        Ordering::Equal
    } else if rpo_right_greater(u, v, key) {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

/// RPO with the first letter as the outermost symbol.
fn rpo_left_greater<F>(u: &[usize], v: &[usize], key: &F) -> bool
where
    F: Fn(usize) -> (u64, usize),
{
    if v.is_empty() {
        return !u.is_empty();
    }
    if u.is_empty() {
        return false;
    }
    let ua = &u[1..];
    let vb = &v[1..];
    if rpo_left_geq(ua, v, key) {
        return true;
    }
    let a = key(u[0]);
    let b = key(v[0]);
    let prec = a > b || (a == b && rpo_left_greater(ua, vb, key));
    prec && rpo_left_greater(u, vb, key)
}

fn rpo_left_geq<F>(u: &[usize], v: &[usize], key: &F) -> bool
where
    F: Fn(usize) -> (u64, usize),
{
    u == v || rpo_left_greater(u, v, key)
}

fn rpo_left_cmp<F>(u: &[usize], v: &[usize], key: &F) -> Ordering
where
    F: Fn(usize) -> (u64, usize),
{
    if u == v {
        Ordering::Equal
    } else if rpo_left_greater(u, v, key) {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}

fn recursive_cmp(u: &[usize], v: &[usize]) -> Ordering {
    rpo_right_cmp(u, v, &|g: usize| (0u64, g))
}

fn rtrecursive_cmp(u: &[usize], v: &[usize]) -> Ordering {
    rpo_left_cmp(u, v, &|g: usize| (0u64, g))
}

fn wreath_cmp(u: &[usize], v: &[usize], levels: &[u64]) -> Ordering {
    rpo_right_cmp(u, v, &|g: usize| (levels.get(g).copied().unwrap_or(0), g))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortlex_length_then_lex() {
        let o = WordOrdering::ShortLex;
        assert_eq!(o.compare(&[0], &[0, 0]), Ordering::Less); // shorter first
        assert_eq!(o.compare(&[0, 1], &[1, 0]), Ordering::Less); // lex within length
        assert_eq!(o.compare(&[1], &[0]), Ordering::Greater); // gen order 0 < 1
        assert_eq!(o.compare(&[0, 1], &[0, 1]), Ordering::Equal);
    }

    #[test]
    fn recursive_higher_gen_dominates() {
        let o = WordOrdering::Recursive;
        // b > a^k for every k (letter 1 dominates letter 0).
        assert_eq!(o.compare(&[1], &[0, 0, 0]), Ordering::Greater);
        assert_eq!(o.compare(&[0, 0, 0], &[1]), Ordering::Less);
        assert_eq!(o.compare(&[0, 0], &[0, 0]), Ordering::Equal);
    }

    #[test]
    fn recursive_is_a_total_order() {
        let o = WordOrdering::Recursive;
        let words: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![1],
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
        ];
        for a in &words {
            for b in &words {
                let ab = o.compare(a, b);
                let ba = o.compare(b, a);
                assert_eq!(ab, ba.reverse(), "antisymmetry {:?} {:?}", a, b);
                if a == b {
                    assert_eq!(ab, Ordering::Equal);
                } else {
                    assert_ne!(ab, Ordering::Equal);
                }
            }
        }
    }

    #[test]
    fn recursive_is_translation_invariant() {
        let o = WordOrdering::Recursive;
        // If u < v then x u y < x v y for all x, y.
        let pairs: Vec<(Vec<usize>, Vec<usize>)> =
            vec![(vec![0, 0, 0], vec![1]), (vec![0, 1], vec![1, 0])];
        for (u, v) in pairs {
            let base = o.compare(&u, &v);
            for x in [vec![], vec![0], vec![1], vec![1, 0]] {
                for y in [vec![], vec![0], vec![1], vec![0, 1]] {
                    let mut xu = x.clone();
                    xu.extend_from_slice(&u);
                    xu.extend_from_slice(&y);
                    let mut xv = x.clone();
                    xv.extend_from_slice(&v);
                    xv.extend_from_slice(&y);
                    assert_eq!(o.compare(&xu, &xv), base, "x={:?} y={:?}", x, y);
                }
            }
        }
    }

    #[test]
    fn wtlex_by_weight() {
        // weight(0)=2, weight(1)=1 => a single '0' outweighs two '1's.
        let o = WordOrdering::WtLex(vec![2, 1]);
        assert_eq!(o.compare(&[0], &[1, 1]), Ordering::Less); // 2 == 2 => shortlex, len 1 < 2
        assert_eq!(o.compare(&[0], &[1]), Ordering::Greater); // 2 > 1
    }

    #[test]
    fn wreath_matches_recursive_for_canonical_levels() {
        let rec = WordOrdering::Recursive;
        let wr = WordOrdering::Wreath(vec![0, 1, 2]);
        let words: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![2, 0], vec![1, 1], vec![2]];
        for a in &words {
            for b in &words {
                assert_eq!(wr.compare(a, b), rec.compare(a, b), "{:?} {:?}", a, b);
            }
        }
    }

    #[test]
    fn orient_puts_larger_on_the_left() {
        let o = WordOrdering::ShortLex;
        assert_eq!(o.orient(&[0, 0], &[0]), Some((vec![0, 0], vec![0])));
        assert_eq!(o.orient(&[0], &[0, 0]), Some((vec![0, 0], vec![0])));
        assert_eq!(o.orient(&[0], &[0]), None);
    }
}
