//! Degree 5: the Frobenius cycle-type sieve over the complete table of the
//! five transitive groups of degree 5, sharpened by an exact ordered-pair
//! resolvent. Decisions happen **only** when the exhibited elements and exact
//! invariants force uniqueness; the one genuinely resolvent-blind pair
//! ({F20, S5}, both 2-transitive with identical ordered-pair signature) is
//! returned as an honest `Unresolved`.
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups): the degree-5 `GaloisGroup`
//!   decision (Stauduhar with the 5T1–5T5 lattice).
//! - SageMath: `sage.rings.number_field.galois_group`.
//!
//! **Transitive groups of degree 5** (Butler–McKay numbering; cycle types and
//! ordered-pair orbit signatures re-verified for this port with sympy's
//! `PermutationGroup` orbit computations):
//!
//! | group | order | cycle types                                        | orbits on ordered pairs |
//! |-------|-------|----------------------------------------------------|-------------------------|
//! | 5T1 C5  | 5   | 1⁵, 5                                              | [5,5,5,5]               |
//! | 5T2 D5  | 10  | 1⁵, 2²1, 5                                         | [10,10]                 |
//! | 5T3 F20 | 20  | 1⁵, 2²1, 41, 5                                     | [20]                    |
//! | 5T4 A5  | 60  | 1⁵, 2²1, 31², 5                                    | [20]                    |
//! | 5T5 S5  | 120 | 1⁵, 21³, 2²1, 31², 32, 41, 5                       | [20]                    |
//!
//! The **ordered-pair resolvent** used here is
//! `R_s(t) = ∏_{i≠j} (t − (α_j − s·α_i))` (default `s = 1`, the difference
//! resolvent), built exactly as `Res_x(f(x), f(t + s·x))` divided by its
//! diagonal `∏_i (t − (1−s)α_i)`. When `R_s` is squarefree, the multiset of its
//! irreducible-factor degrees over `ℚ` equals the orbit-length multiset of
//! `Gal(f)` acting on ordered pairs of distinct roots — which separates
//! C5 ([5,5,5,5]) from D5 ([10,10]) from the 2-transitive groups ([20]).

use crate::sieve::{frobenius_cycle_types, DEFAULT_MAX_GOOD_PRIMES};
use crate::types::{Candidates, Evidence, GaloisGroupResult, GroupId};
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::{bivariate, zassenhaus};
use rustmath_rationals::Rational;

struct QuinticCandidate {
    id: GroupId,
    /// `G ⊆ A₅`?
    even: bool,
    /// Complete set of cycle types occurring in the group (descending parts).
    cycle_types: &'static [&'static [usize]],
    /// Orbit lengths on ordered pairs of distinct points, sorted ascending.
    pair_signature: &'static [usize],
}

fn quintic_table() -> Vec<QuinticCandidate> {
    const T1: &[&[usize]] = &[&[1, 1, 1, 1, 1], &[5]];
    const T2: &[&[usize]] = &[&[1, 1, 1, 1, 1], &[2, 2, 1], &[5]];
    const T3: &[&[usize]] = &[&[1, 1, 1, 1, 1], &[2, 2, 1], &[4, 1], &[5]];
    const T4: &[&[usize]] = &[&[1, 1, 1, 1, 1], &[2, 2, 1], &[3, 1, 1], &[5]];
    const T5: &[&[usize]] = &[
        &[1, 1, 1, 1, 1],
        &[2, 1, 1, 1],
        &[2, 2, 1],
        &[3, 1, 1],
        &[3, 2],
        &[4, 1],
        &[5],
    ];
    vec![
        QuinticCandidate {
            id: GroupId::new(5, 5, "C5", Some(1)),
            even: true,
            cycle_types: T1,
            pair_signature: &[5, 5, 5, 5],
        },
        QuinticCandidate {
            id: GroupId::new(5, 10, "D5", Some(2)),
            even: true,
            cycle_types: T2,
            pair_signature: &[10, 10],
        },
        QuinticCandidate {
            id: GroupId::new(5, 20, "F20", Some(3)),
            even: false,
            cycle_types: T3,
            pair_signature: &[20],
        },
        QuinticCandidate {
            id: GroupId::new(5, 60, "A5", Some(4)),
            even: true,
            cycle_types: T4,
            pair_signature: &[20],
        },
        QuinticCandidate {
            id: GroupId::new(5, 120, "S5", Some(5)),
            even: false,
            cycle_types: T5,
            pair_signature: &[20],
        },
    ]
}

// ------------------------------------------------------------------------- //
// Ordered-pair resolvent
// ------------------------------------------------------------------------- //

fn rq(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn binom_int(k: usize, a: usize) -> Integer {
    if a > k {
        return Integer::zero();
    }
    let mut num = Integer::one();
    let mut den = Integer::one();
    for i in 0..a {
        num = num * Integer::from((k - i) as i64);
        den = den * Integer::from((i + 1) as i64);
    }
    // exact
    let (q, r) = (num.clone() / den.clone(), num % den);
    debug_assert!(r.is_zero());
    q
}

fn deg_q(p: &[Rational]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == rq(0) {
        n -= 1;
    }
    n as i64 - 1
}

/// Exact quotient over `ℚ[t]`; errors if the division is not exact.
fn exact_div_q(a: &[Rational], b: &[Rational]) -> Result<Vec<Rational>> {
    let db = deg_q(b);
    if db < 0 {
        return Err(MathError::DivisionByZero);
    }
    let lcb_inv = b[db as usize].reciprocal()?;
    let mut r = a.to_vec();
    let mut quo = vec![rq(0); (deg_q(a) - db + 1).max(0) as usize];
    while deg_q(&r) >= db {
        let dr = deg_q(&r) as usize;
        let coeff = r[dr].clone() * lcb_inv.clone();
        let shift = dr - db as usize;
        quo[shift] = coeff.clone();
        for i in 0..=db as usize {
            r[shift + i] = r[shift + i].clone() - coeff.clone() * b[i].clone();
        }
    }
    if deg_q(&r) >= 0 {
        return Err(MathError::InvalidOperation(
            "resolvent diagonal division not exact".to_string(),
        ));
    }
    Ok(quo)
}

/// The exact **ordered-pair resolvent** `R_s(t) = ∏_{i≠j} (t − (α_j − s·α_i))`
/// of a monic `f ∈ ℤ[x]` of degree `n`, for an integer `s ≥ 1`. Degree
/// `n(n−1)`, monic, integral. Built as `Res_x(f(x), f(t + s·x))` divided by the
/// diagonal `∏_i (t − (1−s)·α_i) = Σ_k f_k (1−s)^{n−k} t^k`.
fn ordered_pair_resolvent(f: &[Integer], s: i64) -> Result<Vec<Integer>> {
    let n = f.len() - 1;
    if n < 2 || !f[n].is_one() {
        return Err(MathError::InvalidArgument(
            "ordered_pair_resolvent needs a monic f of degree ≥ 2".to_string(),
        ));
    }
    // f(x) as a bivariate polynomial constant in t: index [x-power][t-power].
    let fbiv: Vec<Vec<Rational>> = f
        .iter()
        .map(|c| vec![Rational::from_integer(c.clone())])
        .collect();
    // g(x, t) = f(t + s·x): term f_k (t + sx)^k contributes
    // f_k · C(k, a) · s^a to [x^a][t^{k−a}].
    let mut gbiv: Vec<Vec<Rational>> = vec![vec![rq(0); n + 1]; n + 1];
    let sz = Integer::from(s);
    for k in 0..=n {
        if f[k].is_zero() {
            continue;
        }
        let mut spow = Integer::one();
        for a in 0..=k {
            let term = f[k].clone() * binom_int(k, a) * spow.clone();
            gbiv[a][k - a] = gbiv[a][k - a].clone() + Rational::from_integer(term);
            spow = spow * sz.clone();
        }
    }
    // Res_x(f(x), f(t + s·x)) = ∏_i f(t + s·α_i) — degree n² in t, monic.
    let res = bivariate::resultant_in_t(&fbiv, &gbiv);
    // Diagonal ∏_i (t − (1−s)·α_i): coeff of t^k is f_k · (1−s)^{n−k}.
    let c = Integer::from(1 - s);
    let diag: Vec<Rational> = (0..=n)
        .map(|k| Rational::from_integer(f[k].clone() * c.pow((n - k) as u32)))
        .collect();
    let quo = exact_div_q(&res, &diag)?;
    // Integrality: the coefficients are symmetric integral expressions in the
    // roots of a monic integer polynomial.
    let mut out = Vec::with_capacity(quo.len());
    for c in &quo {
        if !c.is_integer() {
            return Err(MathError::InvalidOperation(
                "ordered-pair resolvent coefficient not integral".to_string(),
            ));
        }
        out.push(c.numerator().clone());
    }
    while out.len() > 1 && out.last().map(|c| c.is_zero()).unwrap_or(false) {
        out.pop();
    }
    Ok(out)
}

/// Factor-degree multiset of the first squarefree ordered-pair resolvent
/// `R_s` for `s = 1, 2, …, 8`. Returns `(s, sorted degrees)`. Squarefreeness
/// (all factor multiplicities 1) certifies that the pair-to-value map is
/// injective, so the degrees are exactly the orbit lengths of `Gal(f)` on
/// ordered pairs of distinct roots.
fn ordered_pair_signature(f: &[Integer]) -> Result<(i64, Vec<usize>)> {
    let n = f.len() - 1;
    for s in 1..=8i64 {
        let r = ordered_pair_resolvent(f, s)?;
        if r.len() != n * (n - 1) + 1 {
            continue; // degenerate (leading cancellation) — try next s
        }
        let (_, factors) = zassenhaus::factor(&r).map_err(|_| {
            MathError::NotSupported("factor recombination limit exceeded".to_string())
        })?;
        if factors.iter().any(|(_, m)| *m > 1) {
            continue; // not squarefree: some α_j − s·α_i collide — try next s
        }
        let mut degs: Vec<usize> = factors
            .iter()
            .filter_map(|(g, _)| (g.len() >= 2).then_some(g.len() - 1))
            .collect();
        degs.sort_unstable();
        return Ok((s, degs));
    }
    Err(MathError::NotSupported(
        "no squarefree ordered-pair resolvent found for s ≤ 8".to_string(),
    ))
}

// ------------------------------------------------------------------------- //
// Decision engine
// ------------------------------------------------------------------------- //

/// Degree 5, monic irreducible. See module docs for the method; every removal
/// of a candidate is certified (parity, an exhibited Frobenius element, or an
/// exact resolvent signature), and only a unique survivor is `Decided`.
pub fn classify_quintic(f: &[Integer], mut ev: Evidence) -> Result<GaloisGroupResult> {
    let disc_sq = ev.disc_is_square.ok_or_else(|| {
        MathError::InvalidOperation("quintic classifier needs disc_is_square".into())
    })?;
    let table = quintic_table();
    let mut ruled_out: Vec<(GroupId, String)> = Vec::new();
    let mut alive: Vec<&QuinticCandidate> = Vec::new();
    for cand in &table {
        if cand.even == disc_sq {
            alive.push(cand);
        } else {
            ruled_out.push((
                cand.id.clone(),
                if disc_sq {
                    format!(
                        "disc is a perfect square (G ⊆ A5) but {} contains odd permutations",
                        cand.id.name
                    )
                } else {
                    format!(
                        "disc is not a perfect square (G ⊄ A5) but {} ⊆ A5",
                        cand.id.name
                    )
                },
            ));
        }
    }

    // Frobenius sieve: every observed type is the cycle type of an element of
    // G, so any candidate lacking it is ruled out.
    let types = frobenius_cycle_types(f, DEFAULT_MAX_GOOD_PRIMES);
    ev.frobenius_types = types.clone();
    for (p, ct) in &types {
        alive.retain(|cand| {
            let has = cand.cycle_types.iter().any(|t| t == &ct.as_slice());
            if !has {
                ruled_out.push((
                    cand.id.clone(),
                    format!(
                        "Frobenius at p={p} has cycle type {ct:?}, which does not occur in {}",
                        cand.id.name
                    ),
                ));
            }
            has
        });
    }

    if alive.len() == 1 {
        ev.notes.push(format!(
            "parity + Frobenius sieve leave exactly one candidate: {}",
            alive[0].id.name
        ));
        return Ok(GaloisGroupResult::Decided { group: alive[0].id.clone(), evidence: ev });
    }
    if alive.is_empty() {
        return Err(MathError::InvalidOperation(
            "internal contradiction: every degree-5 transitive group ruled out \
             (is the input really irreducible and squarefree?)"
                .to_string(),
        ));
    }

    // Exact sharpening: orbit lengths on ordered pairs of roots.
    match ordered_pair_signature(f) {
        Ok((s, sig)) => {
            ev.resolvent_signatures.push((
                format!("ordered-pair resolvent ∏(t − (α_j − {s}·α_i)), i≠j"),
                sig.clone(),
            ));
            alive.retain(|cand| {
                let ok = cand.pair_signature == sig.as_slice();
                if !ok {
                    ruled_out.push((
                        cand.id.clone(),
                        format!(
                            "orbit lengths on ordered root pairs are {sig:?}, but {} acts \
                             with orbit lengths {:?}",
                            cand.id.name, cand.pair_signature
                        ),
                    ));
                }
                ok
            });
            match alive.len() {
                1 => {
                    ev.notes.push(format!(
                        "ordered-pair orbit signature {sig:?} forces {}",
                        alive[0].id.name
                    ));
                    Ok(GaloisGroupResult::Decided {
                        group: alive[0].id.clone(),
                        evidence: ev,
                    })
                }
                0 => Err(MathError::InvalidOperation(
                    "internal contradiction: ordered-pair signature matches no surviving \
                     candidate"
                        .to_string(),
                )),
                _ => {
                    // Only {F20, S5} can survive here (both 2-transitive, both odd).
                    Ok(GaloisGroupResult::Unresolved {
                        candidates: Candidates::Among(
                            alive.iter().map(|c| c.id.clone()).collect(),
                        ),
                        ruled_out,
                        evidence: ev,
                        blocked_on: "F20 vs S5 is invisible to parity, to the cycle-type \
                                     sieve (unless an S5-only type is found), and to the \
                                     ordered-pair resolvent (both are 2-transitive); exact \
                                     separation needs the sextic (Cayley) resolvent for the \
                                     maximal subgroup F20 < S5, or Stauduhar descent — see \
                                     crate::stauduhar"
                            .to_string(),
                    })
                }
            }
        }
        Err(e) => {
            ev.notes
                .push(format!("ordered-pair resolvent unavailable: {e}"));
            Ok(GaloisGroupResult::Unresolved {
                candidates: Candidates::Among(alive.iter().map(|c| c.id.clone()).collect()),
                ruled_out,
                evidence: ev,
                blocked_on: format!("ordered-pair resolvent failed ({e})"),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn difference_resolvent_of_x2_minus_2() {
        // Roots ±√2; ordered differences: ±2√2 → R(t) = t² − 8.
        let r = ordered_pair_resolvent(&iz(&[-2, 0, 1]), 1).unwrap();
        assert_eq!(r, iz(&[-8, 0, 1]));
    }

    #[test]
    fn difference_resolvent_of_x2_minus_x() {
        // f = x² − x (roots 0, 1): differences ±1 → R(t) = t² − 1.
        let r = ordered_pair_resolvent(&iz(&[0, -1, 1]), 1).unwrap();
        assert_eq!(r, iz(&[-1, 0, 1]));
    }

    #[test]
    fn s2_resolvent_of_x2_minus_2() {
        // s = 2: values α_j − 2α_i for i≠j: roots ±3√2 → t² − 18.
        let r = ordered_pair_resolvent(&iz(&[-2, 0, 1]), 2).unwrap();
        assert_eq!(r, iz(&[-18, 0, 1]));
    }

    #[test]
    fn signature_of_lehmer_quintic_is_c5() {
        // x⁵+x⁴−4x³−3x²+3x+1 (Lehmer): Gal = C5; sympy-verified factor degrees
        // of the difference resolvent: [5,5,5,5].
        let f = iz(&[1, 3, -3, -4, 1, 1]);
        let (s, sig) = ordered_pair_signature(&f).unwrap();
        assert_eq!(s, 1);
        assert_eq!(sig, vec![5, 5, 5, 5]);
    }
}
