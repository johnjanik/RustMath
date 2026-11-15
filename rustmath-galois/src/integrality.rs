//! Bounded integrality from a residue, and the integrality dichotomy between candidates.
//!
//! MAGMA: ch. 38 `IsInt(x, B, S)` ("decides if there is `y` in `Z` with `y = x` up to the
//! precision of `x` and `|y| < B`") and `Bound`; digest
//! `rustmath-lava/docs/Chapter_38_Galois_Theory_of_Number_Fields.md`.
//!
//! Mathematics: K. Geissler, J. Klueners, *Galois group computation for rational polynomials*,
//! J. Symbolic Comput. 30 (2000) 653-674 (bounded integrality of invariants evaluated at `p`-adic
//! roots); R. P. Stauduhar, *The determination of Galois groups*, Math. Comp. 27 (1973) 981-996.
//! The dichotomy is the decision rule of Proposition 5.1 of the paper `golay_m23.tex` of the
//! owner's M23 repository (`d05475b`), used by [`crate::steiner_labelling`] and, later, by the
//! WP-A block-system choice (`docs/workstreams/m23_golay/HANDOFF.md`, 5.3 A4).
//!
//! The fact behind everything here: if `m > 2B`, the class of a residue mod `m` contains at most
//! one integer of absolute value `<= B`, namely its symmetric residue in `(-m/2, m/2]` when that
//! has absolute value `<= B`. So if a quantity is known to be an integer `n` with `|n| <= B`, its
//! residue mod `m` determines `n`. A precision with `m <= 2B` is refused, never used.

use std::fmt;

use rustmath_integers::Integer;

/// The representative of `a` mod `m` in `(-m/2, m/2]`.
///
/// # Panics
///
/// When `m <= 0`.
pub fn symmetric_residue(a: &Integer, m: &Integer) -> Integer {
    assert!(
        m.signum() > 0,
        "symmetric_residue: the modulus {m} is not positive"
    );
    let r = a.modulo(m);
    if &(&r * &Integer::from(2u64)) > m {
        r - m.clone()
    } else {
        r
    }
}

/// A refused precision, or refused candidates of [`integrality_dichotomy`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntegralityError {
    /// `modulus <= 2 bound`: two integers of absolute value `<= bound` can share a residue, so
    /// the residue decides nothing.
    PrecisionTooLow {
        /// The modulus `p^N`.
        modulus: Integer,
        /// The proven bound `B`.
        bound: Integer,
    },
    /// The candidate at `index` has no entry. It would pass vacuously (every entry of nothing is
    /// within the bound), so it is refused.
    EmptyCandidate {
        /// Its position in the list.
        index: usize,
    },
    /// The candidate at `index` has `len` entries, the first one `expected`. Candidates for one
    /// quantity (a value, a coefficient vector) have one length.
    CandidateLengths {
        /// Its position in the list.
        index: usize,
        /// Its length.
        len: usize,
        /// The length of the first candidate.
        expected: usize,
    },
}

impl fmt::Display for IntegralityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrecisionTooLow { modulus, bound } => write!(
                f,
                "precision too low: the modulus {modulus} is not above twice the bound {bound}"
            ),
            Self::EmptyCandidate { index } => write!(f, "candidate {index} has no entry"),
            Self::CandidateLengths {
                index,
                len,
                expected,
            } => write!(
                f,
                "candidate {index} has {len} entries, the first candidate {expected}"
            ),
        }
    }
}

impl std::error::Error for IntegralityError {}

fn check_precision(modulus: &Integer, bound: &Integer) -> Result<(), IntegralityError> {
    if modulus <= &(bound * &Integer::from(2u64)) {
        return Err(IntegralityError::PrecisionTooLow {
            modulus: modulus.clone(),
            bound: bound.clone(),
        });
    }
    Ok(())
}

/// MAGMA `IsInt`: `Ok(Some(n))` when the symmetric residue `n` of `residue` mod `modulus` has
/// `|n| <= bound`; then `n` is the only integer of absolute value `<= bound` in the class
/// (because `modulus > 2 bound`). `Ok(None)` otherwise: the class holds no integer within the
/// bound, so a quantity proven to be such an integer is not this one.
///
/// # Errors
///
/// `PrecisionTooLow` when `modulus <= 2 bound`.
pub fn bounded_integer(
    residue: &Integer,
    modulus: &Integer,
    bound: &Integer,
) -> Result<Option<Integer>, IntegralityError> {
    check_precision(modulus, bound)?;
    let n = symmetric_residue(residue, modulus);
    Ok(if &n.abs() <= bound { Some(n) } else { None })
}

/// Why an [`integrality_dichotomy`] decided nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DichotomyGap {
    /// At least two candidates are integral within the bound. A larger modulus may separate
    /// them (each attempt with `modulus > 2 bound` is a proof on its own).
    SeveralWithinBound,
    /// No candidate is integral within the bound. Then the premise of the caller (one candidate
    /// is such an integer) is false; what that proves is the caller's to say.
    NoneWithinBound,
}

/// The outcome of [`integrality_dichotomy`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dichotomy<L> {
    /// Exactly one candidate is integral within the bound: it is `label`, with the integer
    /// vector `value`. `rejected` lists the other candidates with their symmetric residues.
    Decided {
        /// The label of the unique candidate within the bound.
        label: L,
        /// Its entries, as integers (the symmetric residues).
        value: Vec<Integer>,
        /// The other candidates and their symmetric residues.
        rejected: Vec<(L, Vec<Integer>)>,
    },
    /// Nothing is decided.
    Unresolved {
        /// Why.
        gap: DichotomyGap,
        /// Every candidate with its symmetric residues.
        residues: Vec<(L, Vec<Integer>)>,
    },
}

/// The integrality dichotomy (paper, Proposition 5.1): the caller knows that **exactly one** of
/// the labelled candidates is a vector of integers of absolute value `<= bound` (the true one),
/// and gives the residues mod `modulus` of every candidate. A candidate passes when every entry's
/// symmetric residue has absolute value `<= bound`. If exactly one passes it is the true one
/// (`Decided`), since the true one always passes when `modulus > 2 bound`. Otherwise nothing is
/// decided (`Unresolved`). The vector form serves WP-A A4, where a candidate is the coefficient
/// vector of a polynomial.
///
/// The decision is a proof only relative to the caller's premise; a false premise can give a
/// false `Decided` (design SL, review 1, m1). An empty list of candidates gives
/// `Unresolved { gap: NoneWithinBound }` (the premise fails).
///
/// # Errors
///
/// `PrecisionTooLow` when `modulus <= 2 bound`; `EmptyCandidate` for a candidate with no entry
/// (it would pass vacuously); `CandidateLengths` when two candidates differ in length (review 1
/// of SL, m4).
pub fn integrality_dichotomy<L: Clone>(
    candidates: &[(L, Vec<Integer>)],
    modulus: &Integer,
    bound: &Integer,
) -> Result<Dichotomy<L>, IntegralityError> {
    check_precision(modulus, bound)?;
    let expected = candidates.first().map_or(0, |(_, v)| v.len());
    for (index, (_, v)) in candidates.iter().enumerate() {
        if v.is_empty() {
            return Err(IntegralityError::EmptyCandidate { index });
        }
        if v.len() != expected {
            return Err(IntegralityError::CandidateLengths {
                index,
                len: v.len(),
                expected,
            });
        }
    }
    let residues: Vec<(L, Vec<Integer>)> = candidates
        .iter()
        .map(|(l, v)| {
            (
                l.clone(),
                v.iter().map(|c| symmetric_residue(c, modulus)).collect(),
            )
        })
        .collect();
    let passing: Vec<usize> = residues
        .iter()
        .enumerate()
        .filter(|(_, (_, v))| v.iter().all(|c| &c.abs() <= bound))
        .map(|(i, _)| i)
        .collect();
    match passing.as_slice() {
        [i] => {
            let (label, value) = residues[*i].clone();
            let rejected = residues
                .iter()
                .enumerate()
                .filter(|(j, _)| j != i)
                .map(|(_, r)| r.clone())
                .collect();
            Ok(Dichotomy::Decided {
                label,
                value,
                rejected,
            })
        }
        [] => Ok(Dichotomy::Unresolved {
            gap: DichotomyGap::NoneWithinBound,
            residues,
        }),
        _ => Ok(Dichotomy::Unresolved {
            gap: DichotomyGap::SeveralWithinBound,
            residues,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i(v: i64) -> Integer {
        Integer::from(v)
    }

    #[test]
    fn symmetric_residues() {
        assert_eq!(symmetric_residue(&i(7), &i(5)), i(2));
        assert_eq!(symmetric_residue(&i(8), &i(5)), i(-2));
        assert_eq!(symmetric_residue(&i(-8), &i(5)), i(2));
        // Even modulus: the representative of m/2 is +m/2.
        assert_eq!(symmetric_residue(&i(4), &i(8)), i(4));
        assert_eq!(symmetric_residue(&i(5), &i(8)), i(-3));
        assert_eq!(symmetric_residue(&i(-4), &i(8)), i(4));
        assert_eq!(
            symmetric_residue(&i(1 << 40), &Integer::from(1u64 << 41)),
            i(1 << 40)
        );
        assert_eq!(symmetric_residue(&i(0), &i(1)), i(0));
    }

    #[test]
    fn bounded_integer_needs_m_above_2b() {
        let b = i(10);
        assert_eq!(
            bounded_integer(&i(3), &i(20), &b),
            Err(IntegralityError::PrecisionTooLow {
                modulus: i(20),
                bound: b.clone()
            })
        );
        assert_eq!(bounded_integer(&i(3), &i(21), &b), Ok(Some(i(3))));
        assert_eq!(bounded_integer(&i(11), &i(21), &b), Ok(Some(i(-10))));
        assert_eq!(bounded_integer(&i(10), &i(21), &b), Ok(Some(i(10))));
        assert_eq!(bounded_integer(&i(25), &i(23), &b), Ok(Some(i(2))));
        // 11 and 12 mod 23 are 11 and -11: outside the bound.
        assert_eq!(bounded_integer(&i(11), &i(23), &b), Ok(None));
        assert_eq!(bounded_integer(&i(12), &i(23), &b), Ok(None));
    }

    #[test]
    fn three_outcomes() {
        let m = i(1000);
        let b = i(100);
        let c = |a: i64, bb: i64| vec![("a", vec![i(a)]), ("b", vec![i(bb)])];
        assert_eq!(
            integrality_dichotomy(&c(1005, 400), &m, &b),
            Ok(Dichotomy::Decided {
                label: "a",
                value: vec![i(5)],
                rejected: vec![("b", vec![i(400)])]
            })
        );
        assert_eq!(
            integrality_dichotomy(&c(5, -7), &m, &b),
            Ok(Dichotomy::Unresolved {
                gap: DichotomyGap::SeveralWithinBound,
                residues: c(5, -7)
            })
        );
        assert_eq!(
            integrality_dichotomy(&c(300, 700), &m, &b),
            Ok(Dichotomy::Unresolved {
                gap: DichotomyGap::NoneWithinBound,
                residues: c(300, -300)
            })
        );
        assert!(integrality_dichotomy(&c(1, 2), &i(200), &b).is_err());
        // Vector candidates: every entry must pass.
        let v = vec![("x", vec![i(1), i(2)]), ("y", vec![i(3), i(500)])];
        match integrality_dichotomy(&v, &m, &b).unwrap() {
            Dichotomy::Decided { label, value, .. } => {
                assert_eq!(label, "x");
                assert_eq!(value, vec![i(1), i(2)]);
            }
            other => panic!("expected Decided, got {other:?}"),
        }
    }

    /// Review 1 of SL, m4: an empty candidate passed vacuously and was `Decided`.
    #[test]
    fn malformed_candidates_are_refused() {
        let m = i(1000);
        let b = i(100);
        let empty: Vec<(&str, Vec<Integer>)> = vec![("empty", vec![]), ("big", vec![i(400)])];
        assert_eq!(
            integrality_dichotomy(&empty, &m, &b),
            Err(IntegralityError::EmptyCandidate { index: 0 })
        );
        let empty_last = vec![("a", vec![i(400)]), ("b", vec![])];
        assert_eq!(
            integrality_dichotomy(&empty_last, &m, &b),
            Err(IntegralityError::EmptyCandidate { index: 1 })
        );
        let unequal = vec![("a", vec![i(1)]), ("b", vec![i(2), i(3)])];
        assert_eq!(
            integrality_dichotomy(&unequal, &m, &b),
            Err(IntegralityError::CandidateLengths {
                index: 1,
                len: 2,
                expected: 1
            })
        );
        // No candidate at all: nothing is decided.
        let none: Vec<(&str, Vec<Integer>)> = vec![];
        assert_eq!(
            integrality_dichotomy(&none, &m, &b),
            Ok(Dichotomy::Unresolved {
                gap: DichotomyGap::NoneWithinBound,
                residues: vec![]
            })
        );
    }
}
