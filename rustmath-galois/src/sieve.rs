//! Frobenius cycle-type sieve — the degree-independent engine.
//!
//! References:
//! - MAGMA Handbook, Chapter 38 (Galois Groups), "the cycle types of Frobenius
//!   elements at unramified primes" pre-filter used by `GaloisGroup`.
//! - SageMath: `sage.rings.number_field.galois_group`; the sieve is the
//!   classical Dedekind reduction underlying `galois_group(algorithm=...)`.
//!
//! **Dedekind's theorem.** For monic squarefree `f ∈ ℤ[x]` and a prime `p` such
//! that `f mod p` is squarefree, the degrees of the irreducible factors of
//! `f mod p` are the cycle lengths of the Frobenius conjugacy class at `p` in
//! `Gal(f) ⊆ S_n`. Every observed type therefore certifies an **element of the
//! group** — sound for ruling candidates out and for exhibiting special
//! elements (p-cycles, transpositions, 3-cycles), never for deciding by
//! absence: Chebotarev guarantees every class eventually appears, but a
//! bounded search that fails to see a type proves nothing.
//!
//! For degrees with no built-in transitive-group table (n ≥ 6) the sieve still
//! decides the two ubiquitous cases by *exhibiting* elements (Jordan's
//! criteria, see [`classify_general`]); everything else is returned as an
//! honest [`GaloisGroupResult::Unresolved`].

use crate::types::{
    factorial_u128, Candidates, CycleType, Evidence, GaloisGroupResult, GroupId,
};
use rustmath_core::{MathError, Result};
use rustmath_integers::Integer;
use rustmath_polynomials::fp_factor;

/// Default number of good primes the sieve visits (in increasing order,
/// starting at 2). Deterministic, so results are reproducible.
pub const DEFAULT_MAX_GOOD_PRIMES: usize = 100;

/// Hard cap on the prime search; reached only for inputs whose discriminant
/// kills nearly every small prime.
const PRIME_SEARCH_CAP: u64 = 1_000_000;

fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Reduce monic `f ∈ ℤ[x]` mod `p`; `None` if the reduction is not squarefree
/// (a "bad"/ramified prime for the sieve).
fn reduce_squarefree_mod_p(f: &[Integer], p: u64) -> Option<Vec<i64>> {
    let pz = Integer::from(p as i64);
    let fp: Vec<i64> = f.iter().map(|c| c.modulo(&pz).to_i64()).collect();
    let fp = fp_factor::trim(&fp);
    // f is monic, so the degree is preserved for every p.
    if fp_factor::degree(&fp) != (f.len() - 1) as i64 {
        return None;
    }
    let der = fp_factor::derivative_of(&fp, p as i64);
    if fp_factor::is_zero(&der) {
        return None;
    }
    let g = fp_factor::gcd(&fp, &der, p as i64);
    if fp_factor::degree(&g) != 0 {
        return None;
    }
    Some(fp)
}

/// Cycle type of the Frobenius element at `p` (Dedekind): the multiset of
/// irreducible-factor degrees of `f mod p`, sorted descending. `None` when `p`
/// is a bad prime (`f mod p` not squarefree).
pub fn cycle_type_mod_p(f: &[Integer], p: u64) -> Option<CycleType> {
    let fp = reduce_squarefree_mod_p(f, p)?;
    let mut ct: CycleType = Vec::new();
    for (d, g) in fp_factor::distinct_degree_factor(&fp, p as i64) {
        let count = fp_factor::degree(&g) / d;
        for _ in 0..count {
            ct.push(d as usize);
        }
    }
    ct.sort_unstable_by(|a, b| b.cmp(a));
    // Defensive: the factor degrees of a squarefree reduction must sum to n.
    if ct.iter().sum::<usize>() == f.len() - 1 {
        Some(ct)
    } else {
        None
    }
}

/// Collect the **distinct** Frobenius cycle types observed over the first
/// `max_good_primes` good primes, each paired with its smallest witness prime.
/// Monic squarefree `f` required.
pub fn frobenius_cycle_types(f: &[Integer], max_good_primes: usize) -> Vec<(u64, CycleType)> {
    let mut found: Vec<(u64, CycleType)> = Vec::new();
    let mut good = 0usize;
    let mut p = 2u64;
    while good < max_good_primes && p <= PRIME_SEARCH_CAP {
        if is_prime_u64(p) {
            if let Some(ct) = cycle_type_mod_p(f, p) {
                good += 1;
                if !found.iter().any(|(_, t)| *t == ct) {
                    found.push((p, ct));
                }
            }
        }
        p += 1;
    }
    found
}

/// Does a permutation of cycle type `ct` have a power that is a **pure
/// `q`-cycle** (`q` prime)? True iff exactly one part equals `q` and no other
/// part is divisible by `q`: raising to `L = lcm(other parts)` then kills every
/// other cycle while the `q`-cycle survives (`q ∤ L`).
pub fn power_gives_pure_cycle(ct: &[usize], q: usize) -> bool {
    ct.iter().filter(|&&x| x == q).count() == 1
        && ct.iter().filter(|&&x| x != q).all(|&x| x % q != 0)
}

/// Is this cycle type an `(n−1, 1)`-cycle, i.e. an `(n−1)`-cycle fixing one
/// point? For transitive `G` this certifies 2-transitivity (the stabilizer of
/// the fixed point acts transitively on the rest).
pub fn is_n_minus_one_cycle(ct: &[usize], n: usize) -> bool {
    n >= 3 && ct.len() == 2 && ct[0] == n - 1 && ct[1] == 1
}

/// General-degree classification for a monic irreducible `f` of degree `n ≥ 6`
/// (also mathematically valid for smaller `n`, where the dedicated tables are
/// used instead). Decides only `S_n` / `A_n`, and only by *exhibiting*
/// elements:
///
/// 1. **Jordan p-cycle criterion.** If a Frobenius power is a pure `q`-cycle
///    for a prime `q` with `n/2 < q ≤ n − 3`, then: transitive + `q`-cycle with
///    `q > n/2` forces primitivity (a block of size `b`, `1 < b < n`, can
///    neither contain the `q` moved points, `q > n/2 ≥ b`, nor be moved in a
///    `q`-orbit of blocks, `q > n/2 ≥ n/b`), and a primitive group containing a
///    `q`-cycle with `q ≤ n − 3` contains `A_n` (Jordan 1873; Wielandt,
///    *Finite Permutation Groups*, Thm. 13.9).
/// 2. **2-transitive + 3-cycle / transposition.** An observed `(n−1,1)` type
///    certifies 2-transitivity, hence primitivity. A primitive group containing
///    a 3-cycle contains `A_n`; one containing a transposition is `S_n`
///    (Jordan; Wielandt Thm. 13.3).
///
/// Parity (discriminant square test) then separates `A_n` from `S_n`.
/// Everything else returns `Unresolved` with the full evidence.
pub fn classify_general(f: &[Integer], mut ev: Evidence) -> Result<GaloisGroupResult> {
    let n = f.len() - 1;
    let disc_sq = ev.disc_is_square == Some(true);
    let types = frobenius_cycle_types(f, DEFAULT_MAX_GOOD_PRIMES);
    ev.frobenius_types = types.clone();

    let mut contains_an: Option<String> = None; // proof that G ⊇ A_n
    let mut is_sn: Option<String> = None; // proof that G = S_n

    // (1) Jordan p-cycle criterion.
    'jordan: for q in (n / 2 + 1)..=n.saturating_sub(3) {
        if !is_prime_u64(q as u64) {
            continue;
        }
        for (p, ct) in &types {
            if power_gives_pure_cycle(ct, q) {
                contains_an = Some(format!(
                    "Frobenius at p={p} has cycle type {ct:?}; its power is a pure {q}-cycle \
                     with n/2 < {q} ≤ n−3: transitivity + a {q}-cycle ({q} > n/2) forces \
                     primitivity, and Jordan's theorem then gives G ⊇ A_{n}"
                ));
                break 'jordan;
            }
        }
    }

    // (2) 2-transitivity witness + 3-cycle / transposition.
    if contains_an.is_none() {
        if let Some((p2t, _)) = types.iter().find(|(_, ct)| is_n_minus_one_cycle(ct, n)) {
            let two_transitive = format!(
                "Frobenius at p={p2t} is an ({}, 1)-cycle: for transitive G the point \
                 stabilizer acts transitively on the remaining {} points, so G is \
                 2-transitive, hence primitive",
                n - 1,
                n - 1
            );
            if let Some((p3, ct3)) = types.iter().find(|(_, ct)| power_gives_pure_cycle(ct, 3)) {
                contains_an = Some(format!(
                    "{two_transitive}; Frobenius at p={p3} (type {ct3:?}) powers to a 3-cycle, \
                     and a primitive group containing a 3-cycle contains A_{n} (Jordan)"
                ));
            } else if let Some((pt, ctt)) =
                types.iter().find(|(_, ct)| power_gives_pure_cycle(ct, 2))
            {
                is_sn = Some(format!(
                    "{two_transitive}; Frobenius at p={pt} (type {ctt:?}) powers to a \
                     transposition, and a primitive group containing a transposition is S_{n} \
                     (Jordan)"
                ));
            }
        }
    }

    let sn = || -> GroupId {
        GroupId { degree: n, order: factorial_u128(n), name: format!("S{n}"), t_number: None }
    };
    let an = || -> GroupId {
        GroupId {
            degree: n,
            order: factorial_u128(n).map(|o| o / 2),
            name: format!("A{n}"),
            t_number: None,
        }
    };

    if let Some(proof) = is_sn {
        if disc_sq {
            return Err(MathError::InvalidOperation(format!(
                "internal contradiction: proof of S_{n} ({proof}) but square discriminant \
                 (G ⊆ A_{n}); this indicates a bug or an invalid (non-squarefree) input"
            )));
        }
        ev.notes.push(proof);
        return Ok(GaloisGroupResult::Decided { group: sn(), evidence: ev });
    }
    if let Some(proof) = contains_an {
        ev.notes.push(proof);
        let group = if disc_sq {
            ev.notes.push(format!(
                "disc is a perfect square, so G ⊆ A_{n}; with G ⊇ A_{n} this decides A_{n}"
            ));
            an()
        } else {
            ev.notes.push(format!(
                "disc is not a perfect square, so G ⊄ A_{n}; with G ⊇ A_{n} this decides S_{n}"
            ));
            sn()
        };
        return Ok(GaloisGroupResult::Decided { group, evidence: ev });
    }

    // Honest refusal: nothing found that forces the group.
    let mut ruled_out = Vec::new();
    if disc_sq {
        ruled_out.push((
            sn(),
            format!("disc is a perfect square, so G ⊆ A_{n} and G ≠ S_{n}"),
        ));
    } else {
        ruled_out.push((
            an(),
            format!("disc is not a perfect square, so G ⊄ A_{n} (G contains odd permutations)"),
        ));
    }
    Ok(GaloisGroupResult::Unresolved {
        candidates: Candidates::Unknown {
            degree: n,
            transitive: true,
            contained_in_alternating: Some(disc_sq),
        },
        ruled_out,
        evidence: ev,
        blocked_on: format!(
            "no transitive-group table built in for degree {n} and no A_{n}/S_{n} certificate \
             was exhibited within {DEFAULT_MAX_GOOD_PRIMES} good primes; completion requires \
             Stauduhar descent through the degree-{n} transitive lattice (see crate::stauduhar)"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn cycle_type_matches_dedekind_for_x5_minus_x_minus_1() {
        // Verified with sympy: x^5−x−1 mod 2 factors as (deg 2)(deg 3) → type (3,2);
        // mod 3 it is irreducible → type (5).
        let f = iz(&[-1, -1, 0, 0, 0, 1]);
        assert_eq!(cycle_type_mod_p(&f, 2), Some(vec![3, 2]));
        assert_eq!(cycle_type_mod_p(&f, 3), Some(vec![5]));
        // p = 163 gives type (2,1,1,1) (first transposition witness; sympy-verified).
        assert_eq!(cycle_type_mod_p(&f, 163), Some(vec![2, 1, 1, 1]));
    }

    #[test]
    fn bad_primes_are_rejected() {
        // f = x² − 2: 2 divides disc = 8, so x² mod 2 is not squarefree.
        let f = iz(&[-2, 0, 1]);
        assert_eq!(cycle_type_mod_p(&f, 2), None);
        // 7² = 49 ≡ 2 mod 47? 7²=49≡2 (mod 47): x²−2 splits mod 47 → (1,1).
        assert_eq!(cycle_type_mod_p(&f, 47), Some(vec![1, 1]));
    }

    #[test]
    fn pure_cycle_extraction() {
        assert!(power_gives_pure_cycle(&[5, 3], 5)); // lcm(3)=3, 5∤3
        assert!(power_gives_pure_cycle(&[3, 2, 1], 2)); // (3,2,1)^3 = transposition
        assert!(power_gives_pure_cycle(&[3, 2, 2, 1], 3)); // ^2 = 3-cycle
        assert!(!power_gives_pure_cycle(&[2, 2, 1], 2)); // two 2-parts
        assert!(!power_gives_pure_cycle(&[6, 3], 3)); // 3 | 6
        assert!(!power_gives_pure_cycle(&[5, 5], 5)); // two 5-parts
    }
}
