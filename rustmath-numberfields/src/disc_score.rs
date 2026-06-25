//! Discriminant scorer for the CFT construction stack.
//!
//! Ported from `frobenius/disc_score.sage` (the "small-disc strategy" scorer,
//! family 3 / Entry 39). That Sage module ranks candidate abelian/relative
//! extensions `L/K` by the **exact** size of the field discriminant `|D_L|`,
//! using the conductor–discriminant relation rather than constructing the
//! (large, expensive) absolute field.
//!
//! # The formula
//!
//! For a relative extension `L/K` of degree `[L:K] = m`, the tower/conductor–
//! discriminant relation gives the absolute discriminant of `L` as
//!
//! ```text
//!   |D_L| = |D_K|^m · N(d_{L/K}),
//! ```
//!
//! where `d_{L/K}` is the relative-discriminant ideal of `L/K` and `N` is its
//! absolute norm. Taking logs,
//!
//! ```text
//!   log|D_L| = m · log|D_K| + log N(d_{L/K}).
//! ```
//!
//! The Sage original specialises to the block-2 relative **quadratic** case
//! (`m = 2`), where `disc_log10(K, DK, gamma) = 2·log10|D_K| + log10 N(d_{L/K})`.
//! For an **abelian** `L/K`, the conductor–discriminant theorem expresses
//! `d_{L/K}` as the product of the conductors of the characters of
//! `Gal(L/K)`:
//!
//! ```text
//!   d_{L/K} = ∏_{χ ∈ Gal(L/K)^} f(χ),   so
//!   log N(d_{L/K}) = Σ_χ log N(f(χ)).
//! ```
//!
//! This module provides:
//! * [`modulus_score`] — the simple ranking key `2·ln|D_K| + ln N(cond)`
//!   (the `m = 2` specialisation, the one the construction driver ranks on);
//! * [`disc_from_conductors`] — the full conductor–discriminant prediction of
//!   `log|D_L|` from the per-character conductor norms;
//! * [`rank_moduli`] — ranks candidate moduli ascending by [`modulus_score`].
//!
//! # Units
//!
//! All scores are returned as **natural logarithms** (nats). This matches the
//! `2·log|D_K| + log N(cond)` spec for this crate. Note the Sage original used
//! `log10` (base-10 "digits"); base only rescales every score by the same
//! constant `ln 10`, so the **ranking order is identical**. Callers wanting
//! base-10 digit counts divide by `std::f64::consts::LN_10`.

use rustmath_integers::Integer;

/// Natural log of the (absolute value of a) nonzero integer `n`.
///
/// Uses `to_f64` when `n` fits in an `f64`; for integers too large for `f64`
/// (where `to_f64` returns `None` or a non-finite value) it falls back to a
/// bit-length estimate `bits · ln 2`, which keeps `ln N` finite and monotone
/// for the very large norms that arise in degree-24 constructions.
fn ln_norm(n: &Integer) -> f64 {
    let a = n.abs();
    if a.is_zero() {
        // log of a zero norm is undefined; treat as -inf so it never "wins"
        // a minimisation. In practice conductor norms are >= 1.
        return f64::NEG_INFINITY;
    }
    if a.is_one() {
        return 0.0;
    }
    match a.to_f64() {
        Some(x) if x.is_finite() && x > 0.0 => x.ln(),
        // Overflowed f64: estimate from the bit length. For x with b bits,
        // 2^(b-1) <= x < 2^b, so ln x ≈ (b - 0.5) · ln 2 to centre the bound.
        _ => {
            let bits = a.bit_length() as f64;
            (bits - 0.5) * std::f64::consts::LN_2
        }
    }
}

/// Simple modulus score `2·ln|D_K| + ln N(conductor)` (natural log, nats).
///
/// This is the `[L:K] = 2` specialisation of the conductor–discriminant
/// relation: `ln|D_L| = 2·ln|D_K| + ln N(d_{L/K})`, with `N(conductor)` taking
/// the role of `N(d_{L/K})`. It is the ranking key the construction driver
/// uses to choose the conductor that minimises `disc(L)`.
///
/// `log_abs_dk` must already be the natural log of `|D_K|` (i.e. `ln|D_K|`);
/// passing it in avoids recomputing the base-field discriminant per candidate.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_numberfields::disc_score::modulus_score;
///
/// let ln_dk = (1e13_f64).ln();
/// // conductor norm 1 contributes nothing:
/// let s = modulus_score(ln_dk, &Integer::from(1i64));
/// assert!((s - 2.0 * ln_dk).abs() < 1e-9);
/// ```
pub fn modulus_score(log_abs_dk: f64, conductor_norm: &Integer) -> f64 {
    2.0 * log_abs_dk + ln_norm(conductor_norm)
}

/// Full conductor–discriminant prediction of `ln|D_L|` (natural log, nats).
///
/// Given `ln|D_K|` and the list of absolute norms `N(f(χ))` of the conductors
/// of the characters `χ` of the abelian group `Gal(L/K)`, returns
///
/// ```text
///   ln|D_L| = [L:K]·ln|D_K| + Σ_χ ln N(f(χ)),
/// ```
///
/// where `[L:K]` equals the number of characters supplied (`= |Gal(L/K)|`).
/// The trivial character contributes conductor norm `1` (`ln 1 = 0`); callers
/// may include or omit it freely. The conductor product `∏_χ f(χ) = d_{L/K}`
/// is exactly the conductor–discriminant theorem.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_numberfields::disc_score::disc_from_conductors;
///
/// // Cyclic cubic over Q of conductor f = 7: characters {1, 7, 7},
/// // |D_K| = 1, so |D_L| = 7^2 = 49 and ln|D_L| = ln 49.
/// let norms = [Integer::from(1i64), Integer::from(7i64), Integer::from(7i64)];
/// let ln_dl = disc_from_conductors(0.0, &norms);
/// assert!((ln_dl - 49.0_f64.ln()).abs() < 1e-9);
/// ```
pub fn disc_from_conductors(log_abs_dk: f64, char_conductor_norms: &[Integer]) -> f64 {
    let m = char_conductor_norms.len() as f64; // [L:K] = number of characters
    let sum_ln: f64 = char_conductor_norms.iter().map(ln_norm).sum();
    m * log_abs_dk + sum_ln
}

/// Rank candidate moduli ascending by [`modulus_score`].
///
/// Each candidate is `(id, conductor_norm)`. Returns `(id, score)` pairs sorted
/// by increasing score (smallest predicted discriminant first), preserving the
/// natural-log units of [`modulus_score`]. Ties keep their relative input order
/// (the sort is stable). NaN scores (which cannot arise from finite norms) sort
/// to the end.
///
/// # Examples
///
/// ```
/// use rustmath_integers::Integer;
/// use rustmath_numberfields::disc_score::rank_moduli;
///
/// let ln_dk = 1.0;
/// let cands = [(0usize, Integer::from(13i64)), (1, Integer::from(2i64))];
/// let ranked = rank_moduli(ln_dk, &cands);
/// assert_eq!(ranked[0].0, 1); // norm 2 scores lower than norm 13
/// ```
pub fn rank_moduli(log_abs_dk: f64, candidates: &[(usize, Integer)]) -> Vec<(usize, f64)> {
    let mut scored: Vec<(usize, f64)> = candidates
        .iter()
        .map(|(id, norm)| (*id, modulus_score(log_abs_dk, norm)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn modulus_score_conductor_one() {
        // |D_K| = 10^13, conductor norm 1 -> 2·ln(10^13).
        let ln_dk = 1e13_f64.ln();
        let s = modulus_score(ln_dk, &int(1));
        let expected = 2.0 * 1e13_f64.ln();
        assert!((s - expected).abs() < 1e-9, "got {s}, want {expected}");
    }

    #[test]
    fn modulus_score_conductor_seven() {
        // conductor norm 7 -> 2·ln(10^13) + ln 7.
        let ln_dk = 1e13_f64.ln();
        let s = modulus_score(ln_dk, &int(7));
        let expected = 2.0 * 1e13_f64.ln() + 7.0_f64.ln();
        assert!((s - expected).abs() < 1e-9, "got {s}, want {expected}");
    }

    #[test]
    fn modulus_score_is_additive_in_conductor() {
        // score(7) - score(1) == ln 7 exactly (independent of D_K).
        let ln_dk = 42.0;
        let d = modulus_score(ln_dk, &int(7)) - modulus_score(ln_dk, &int(1));
        assert!((d - 7.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn rank_moduli_orders_ascending() {
        let ln_dk = 5.0;
        // ids deliberately shuffled relative to norm order.
        let cands = [
            (10usize, int(101)),
            (20, int(2)),
            (30, int(50)),
            (40, int(7)),
        ];
        let ranked = rank_moduli(ln_dk, &cands);
        let ids: Vec<usize> = ranked.iter().map(|(id, _)| *id).collect();
        // ascending by norm: 2 (id20), 7 (id40), 50 (id30), 101 (id10)
        assert_eq!(ids, vec![20, 40, 30, 10]);
        // scores must be non-decreasing
        for w in ranked.windows(2) {
            assert!(w[0].1 <= w[1].1, "not sorted ascending: {ranked:?}");
        }
        // and the lowest score equals the formula for norm 2
        assert!((ranked[0].1 - (2.0 * ln_dk + 2.0_f64.ln())).abs() < 1e-12);
    }

    #[test]
    fn rank_moduli_stable_on_ties() {
        let ln_dk = 0.0;
        let cands = [(1usize, int(5)), (2, int(5)), (3, int(5))];
        let ranked = rank_moduli(ln_dk, &cands);
        let ids: Vec<usize> = ranked.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn disc_from_conductors_cyclic_cubic() {
        // Cyclic cubic over Q of conductor f: characters {1, f, f}, |D_K| = 1.
        // Conductor–discriminant: disc = 1·f·f = f^2. Test f = 7 -> disc 49.
        let norms = [int(1), int(7), int(7)];
        let ln_dl = disc_from_conductors(0.0, &norms);
        assert!((ln_dl - 49.0_f64.ln()).abs() < 1e-9, "got {ln_dl}");

        // f = 9 (conductor of the cubic in Q(zeta_9)^+): disc = 81.
        let norms9 = [int(1), int(9), int(9)];
        let ln_dl9 = disc_from_conductors(0.0, &norms9);
        assert!((ln_dl9 - 81.0_f64.ln()).abs() < 1e-9, "got {ln_dl9}");
    }

    #[test]
    fn disc_from_conductors_includes_base_field_power() {
        // [L:K] = number of characters; ln|D_L| = m·ln|D_K| + Σ ln N(f).
        // 3 characters over a base with ln|D_K| = 2.0, conductors {1, p, p}, N(p)=11.
        let ln_dk = 2.0;
        let norms = [int(1), int(11), int(11)];
        let ln_dl = disc_from_conductors(ln_dk, &norms);
        let expected = 3.0 * ln_dk + 2.0 * 11.0_f64.ln();
        assert!((ln_dl - expected).abs() < 1e-9, "got {ln_dl}, want {expected}");
    }

    #[test]
    fn modulus_score_matches_disc_from_conductors_quadratic() {
        // The m=2 modulus_score is exactly disc_from_conductors with characters
        // {trivial, the quadratic character of conductor c}.
        let ln_dk = 3.5;
        let c = int(13);
        let via_modulus = modulus_score(ln_dk, &c);
        let via_cond = disc_from_conductors(ln_dk, &[int(1), c.clone()]);
        assert!((via_modulus - via_cond).abs() < 1e-12);
    }

    #[test]
    fn ln_norm_large_integer_fallback() {
        // 2^200 is finite as f64; check ln matches 200·ln2 closely.
        let two_pow_200 = Integer::from(2i64).pow(200);
        let v = super::ln_norm(&two_pow_200);
        let expected = 200.0 * std::f64::consts::LN_2;
        // f64 path is exact here; relative tolerance.
        assert!((v - expected).abs() / expected < 1e-6, "got {v}, want {expected}");
    }
}
