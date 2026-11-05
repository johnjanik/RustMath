//! Linear Feedback Shift Registers (LFSRs) and Berlekamp–Massey synthesis.
//!
//! Port of MAGMA Handbook **Chapter 158 — Pseudo-random Bit Sequences**,
//! §158.2 *Linear Feedback Shift Registers*.
//!
//! An LFSR of length `L` has an initial state `s_0, …, s_{L-1}` of finite-field
//! elements and a *connection polynomial*
//! `C(D) = 1 + c_1 D + c_2 D^2 + … + c_L D^L`. The sequence is extended by the
//! linear recurrence
//!
//! ```text
//!     s_j = - Σ_{i=1}^{L} c_i · s_{j-i}      (j ≥ L).
//! ```
//!
//! Following the foundation notes for this chapter, sequences and connection
//! polynomials are plain `Vec` / slices of field elements (`F: Field` from
//! `rustmath-core`, e.g. `rustmath_finitefields::PrimeField`). No new trait is
//! required: the field element type already carries its own modulus/parent, so
//! neutral elements are derived from existing elements rather than the (parent-less)
//! `Ring::zero()` / `Ring::one()` constructors.
//!
//! The connection polynomial is stored as its coefficient vector
//! `[c_0, c_1, …, c_d]` with `c_0 = 1` (MAGMA requires the constant coefficient
//! of `C(D)` to be 1). Missing high-order coefficients are treated as zero, which
//! lets Berlekamp–Massey return a *singular* LFSR (`deg C(D) < L`): the caller
//! regenerates using the first `L` sequence elements.

use rustmath_core::Field;

/// Additive identity of the field, derived from an existing element `proto`
/// (avoids the parent-less `Ring::zero()`, which panics for `PrimeField`).
#[inline]
fn zero_like<F: Field>(proto: &F) -> F {
    proto.clone() - proto.clone()
}

/// Multiplicative identity of the field, derived from a **nonzero** element
/// `proto` as `proto / proto`.
#[inline]
fn one_like<F: Field>(proto: &F) -> F {
    proto.clone() / proto.clone()
}

/// Read coefficient `c_i` of the connection polynomial, treating any index at or
/// beyond the stored length as zero.
#[inline]
fn conn_coeff<F: Field>(c: &[F], i: usize, zero: &F) -> F {
    if i < c.len() {
        c[i].clone()
    } else {
        zero.clone()
    }
}

/// Compute the next state of the LFSR with connection polynomial `c` and current
/// `state` (MAGMA `LFSRStep(C, S)`).
///
/// The LFSR length `L` is `state.len()`; the returned state drops the oldest
/// element and appends the newly generated one. `c[0]` must be the field's one.
///
/// # Panics
/// Panics if `state` or `c` is empty (no valid recurrence).
pub fn lfsr_step<F: Field>(c: &[F], state: &[F]) -> Vec<F> {
    assert!(!state.is_empty(), "LFSR state must be non-empty");
    assert!(!c.is_empty(), "connection polynomial must be non-empty");
    let l = state.len();
    let zero = zero_like(&state[0]);
    // s_new = - Σ_{i=1}^{L} c_i · s_{L-i}   (state[L-i] = s_{j-i})
    let mut acc = zero.clone();
    for i in 1..=l {
        let ci = conn_coeff(c, i, &zero);
        acc = acc + ci * state[l - i].clone();
    }
    let s_new = -acc;
    let mut next = Vec::with_capacity(l);
    next.extend_from_slice(&state[1..]);
    next.push(s_new);
    next
}

/// Compute the first `t` elements of the LFSR with connection polynomial `c` and
/// initial state `s` (MAGMA `LFSRSequence(C, S, t)`).
///
/// The LFSR length is `s.len()`; the first `min(t, s.len())` output elements are
/// the initial state itself. `c[0]` must be the field's one and `c.len()` may be
/// anywhere from `1` to `s.len() + 1` (higher coefficients are taken as zero).
pub fn lfsr_sequence<F: Field>(c: &[F], s: &[F], t: usize) -> Vec<F> {
    assert!(!s.is_empty(), "initial state must be non-empty");
    assert!(!c.is_empty(), "connection polynomial must be non-empty");
    let l = s.len();
    let zero = zero_like(&s[0]);
    let mut seq: Vec<F> = Vec::with_capacity(t);
    for i in 0..t {
        if i < l {
            seq.push(s[i].clone());
        } else {
            let mut acc = zero.clone();
            for k in 1..=l {
                let ck = conn_coeff(c, k, &zero);
                acc = acc + ck * seq[i - k].clone();
            }
            seq.push(-acc);
        }
    }
    seq
}

/// The Berlekamp–Massey algorithm (MAGMA `BerlekampMassey` /
/// `ConnectionPolynomial` / `CharacteristicPolynomial`).
///
/// Given a sequence `s` over a finite field `F`, returns the connection
/// polynomial `C(D) = [c_0, c_1, …]` (with `c_0 = 1`) and the linear complexity
/// `L` of a shortest LFSR that generates `s`.
///
/// The returned polynomial may have degree strictly less than `L` (a *singular*
/// LFSR); regenerate the sequence with [`lfsr_sequence`] using the **first `L`**
/// elements of `s`.
///
/// For an all-zero (or empty) sequence the minimal LFSR has length `0`; this is
/// reported as `(vec![], 0)`, where the empty vector denotes `C(D) = 1`.
pub fn berlekamp_massey<F: Field>(s: &[F]) -> (Vec<F>, usize) {
    // A prototype nonzero element lets us build the field's 0 and 1.
    let proto = match s.iter().find(|x| !x.is_zero()) {
        Some(p) => p,
        None => return (Vec::new(), 0),
    };
    let zero = zero_like(proto);
    let one = one_like(proto);

    let n = s.len();
    let mut c: Vec<F> = vec![one.clone()]; // current connection polynomial
    let mut b: Vec<F> = vec![one.clone()]; // last C before a length change
    let mut l: usize = 0; // current linear complexity
    let mut m: usize = 1; // steps since last length change
    let mut b_disc: F = one; // last nonzero discrepancy

    for i in 0..n {
        // Discrepancy d = s[i] + Σ_{j=1}^{L} c_j · s[i-j].
        let mut d = s[i].clone();
        for j in 1..=l {
            if j < c.len() {
                d = d + c[j].clone() * s[i - j].clone();
            }
        }

        if d.is_zero() {
            m += 1;
            continue;
        }

        // Save C(x) as it stands *before* this update (becomes the new B on a
        // length change — this must be the pre-update polynomial).
        let prev_c = c.clone();
        let coeff = d.clone() / b_disc.clone();
        // Ensure C is long enough to absorb x^m · B.
        let needed = b.len() + m;
        if c.len() < needed {
            c.resize(needed, zero.clone());
        }
        // C(x) -= coeff · x^m · B(x)
        let update: Vec<F> = b.iter().map(|bk| coeff.clone() * bk.clone()).collect();
        for (k, u) in update.into_iter().enumerate() {
            let idx = k + m;
            c[idx] = c[idx].clone() - u;
        }

        if 2 * l <= i {
            l = i + 1 - l;
            b = prev_c;
            b_disc = d;
            m = 1;
        } else {
            m += 1;
        }
    }

    // Trim trailing zeros but keep the constant term.
    while c.len() > 1 && c.last().map(|x| x.is_zero()).unwrap_or(false) {
        c.pop();
    }
    (c, l)
}

/// Alias for [`berlekamp_massey`] matching MAGMA's `ConnectionPolynomial(S)`.
#[inline]
pub fn connection_polynomial<F: Field>(s: &[F]) -> (Vec<F>, usize) {
    berlekamp_massey(s)
}

/// Alias for [`berlekamp_massey`] matching MAGMA's `CharacteristicPolynomial(S)`.
#[inline]
pub fn characteristic_polynomial<F: Field>(s: &[F]) -> (Vec<F>, usize) {
    berlekamp_massey(s)
}

/// Find the smallest period `p > 0` of an (assumed eventually periodic) sequence,
/// i.e. the least `p` with `seq[i] == seq[i + p]` for every valid `i`.
///
/// Requires `seq` to span at least two full periods to be reliable; returns
/// `None` if no period `≤ seq.len() / 2` is found. Useful for confirming that an
/// LFSR built from a primitive connection polynomial has maximal period
/// `q^L − 1` (the analogue of MAGMA's `IsPrimitive` check on the recovered
/// polynomial in example H158E4).
pub fn sequence_period<F: PartialEq>(seq: &[F]) -> Option<usize> {
    let n = seq.len();
    for p in 1..=n / 2 {
        if (0..n - p).all(|i| seq[i] == seq[i + p]) {
            return Some(p);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_finitefields::PrimeField;
    use rustmath_integers::Integer;

    /// Build a GF(p) element.
    fn gf(v: i64, p: i64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn gf2(v: i64) -> PrimeField {
        gf(v, 2)
    }

    fn gf7(v: i64) -> PrimeField {
        gf(v, 7)
    }

    /// Hand-verified GF(2) recurrence: C(D) = 1 + D + D^2, state [1, 0]
    /// produces 1,0,1,1,0,1,1,0,... (period 3).
    #[test]
    fn lfsr_sequence_gf2_basic() {
        let c = vec![gf2(1), gf2(1), gf2(1)];
        let s = vec![gf2(1), gf2(0)];
        let seq = lfsr_sequence(&c, &s, 6);
        let expected = vec![gf2(1), gf2(0), gf2(1), gf2(1), gf2(0), gf2(1)];
        assert_eq!(seq, expected);
    }

    #[test]
    fn lfsr_step_matches_sequence() {
        let c = vec![gf2(1), gf2(1), gf2(1)];
        let s = vec![gf2(1), gf2(0)];
        // First step: state [1,0] -> [0, s2] where s2 = 1.
        let next = lfsr_step(&c, &s);
        assert_eq!(next, vec![gf2(0), gf2(1)]);
    }

    /// MAGMA example H158E1: build a sequence, recover its connection polynomial
    /// and length via Berlekamp–Massey, then regenerate from the first L elements.
    #[test]
    fn h158e1_berlekamp_massey_recover_and_regenerate() {
        let c = vec![gf2(1), gf2(1), gf2(1)]; // 1 + D + D^2
        let s = vec![gf2(1), gf2(0)];
        let seq = lfsr_sequence(&c, &s, 8);

        let (rec_c, l) = berlekamp_massey(&seq);
        assert_eq!(l, 2, "linear complexity should be 2");
        assert_eq!(rec_c, c, "connection polynomial should be 1 + D + D^2");

        // Regenerate from the FIRST L elements (per MAGMA's note).
        let regen = lfsr_sequence(&rec_c, &seq[..l], seq.len());
        assert_eq!(regen, seq);
    }

    /// Berlekamp–Massey over a larger GF(2) LFSR with a longer connection poly.
    #[test]
    fn berlekamp_massey_gf2_degree4() {
        // C(D) = 1 + D + D^4 (a primitive polynomial over GF(2), period 15).
        let c = vec![gf2(1), gf2(1), gf2(0), gf2(0), gf2(1)];
        let s = vec![gf2(1), gf2(0), gf2(0), gf2(0)];
        let seq = lfsr_sequence(&c, &s, 30);

        let (rec_c, l) = berlekamp_massey(&seq);
        assert_eq!(l, 4);
        let regen = lfsr_sequence(&rec_c, &seq[..l], seq.len());
        assert_eq!(regen, seq);

        // Maximal period 2^4 - 1 = 15.
        assert_eq!(sequence_period(&seq), Some(15));
    }

    /// MAGMA example H158E4 (spirit): over GF(7), search for a degree-2 primitive
    /// connection polynomial (LFSR period 7^2 - 1 = 48), generate the sequence,
    /// recover a connection polynomial via Berlekamp–Massey, and confirm the
    /// recovered LFSR is also primitive (period 48).
    #[test]
    fn h158e4_gf7_primitive_degree2() {
        let period = 48usize;
        // Search for [1, c1, c2] with c2 != 0 whose LFSR (from a nonzero state)
        // reaches maximal period 48.
        let mut found: Option<(Vec<PrimeField>, Vec<PrimeField>)> = None;
        'search: for c1 in 0..7i64 {
            for c2 in 1..7i64 {
                let c = vec![gf7(1), gf7(c1), gf7(c2)];
                let s = vec![gf7(1), gf7(0)];
                let seq = lfsr_sequence(&c, &s, 2 * period);
                if sequence_period(&seq) == Some(period) {
                    found = Some((c, s));
                    break 'search;
                }
            }
        }
        let (c, s) = found.expect("a primitive degree-2 connection poly over GF(7) exists");

        let seq = lfsr_sequence(&c, &s, period);
        assert_eq!(seq.len(), period);

        // Recover via Berlekamp–Massey and regenerate.
        let (rec_c, l) = berlekamp_massey(&seq);
        assert_eq!(l, 2, "recovered linear complexity should be 2");
        let regen = lfsr_sequence(&rec_c, &seq[..l], period);
        assert_eq!(regen, seq);

        // Confirm the recovered LFSR is primitive: period 48.
        let long = lfsr_sequence(&rec_c, &seq[..l], 2 * period);
        assert_eq!(sequence_period(&long), Some(period));
    }

    #[test]
    fn berlekamp_massey_all_zero() {
        let s = vec![gf2(0), gf2(0), gf2(0)];
        let (c, l) = berlekamp_massey(&s);
        assert_eq!(l, 0);
        assert!(c.is_empty());
    }
}
