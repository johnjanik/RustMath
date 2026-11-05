//! # Local root numbers of E/Q at the wild primes p = 2 and p = 3
//!
//! The complete case tables of Kraus (p = 3) and Halberstadt (p = 2)
//! (E. Halberstadt, *Signes locaux des courbes elliptiques en 2 et 3*,
//! C. R. Acad. Sci. Paris 326 (1998), 1047–1050), in the explicit
//! tabulated form of O. G. Rizzo, *Average root numbers for a nonconstant
//! family of elliptic curves*, Compositio Math. 136 (2003) 1–23, Tables II
//! (p = 3) and III (p = 2), keyed on the reduced valuation triplet
//!
//! ```text
//! (a, b, c) = (v_p(c₄), v_p(c₆), v_p(Δ)) − k·(4, 6, 12),
//! k = min(⌊v_p(c₄)/4⌋, ⌊v_p(c₆)/6⌋, ⌊v_p(Δ)/12⌋),
//! ```
//!
//! plus congruence side conditions on the p-free parts c₄′, c₆′, Δ′ and on
//! the partially-descaled cofactors (c_{4,2} = c₄/3², c_{4,4} = c₄/3⁴,
//! c_{6,4} = c₆/2⁴, c_{6,7} = c₆/2⁷ of the k-descaled invariants). The
//! joint reduction by k·(4,6,12) is the scaling u = p^k of the model, so
//! the tables apply to ANY integral Weierstrass model, minimal or not.
//!
//! ## Known corrections applied (documented in the literature)
//!
//! The printed tables in Rizzo's paper contain three known errata, all
//! applied here (and all confirmed by the PARI oracle battery below):
//!
//! * Table II, row (≥5, 6, 9): the special condition is the NEGATION
//!   c₆′² + 2 ≢ 3·c_{4,4} (mod 9) of the printed one (recorded by
//!   Bettin–David–Delaunay, arXiv:1612.03095, and Desjardins, JTNB 32
//!   (2020) 73–101);
//! * Table III, second row: the triplet is (0, 0, ≥0), not (0, 0, >0);
//! * Table III, row (2, 3, 1): the Kodaira symbol is I₂*, not I₂.
//!
//! ## Validation (performed BEFORE this port was written)
//!
//! The transcription was validated against PARI/GP (`ellrootno`,
//! `elllocalred` — an independent implementation of the same Halberstadt
//! tables that has been production code for two decades) on **51,212
//! integral models** (random general models, structured short-Weierstrass
//! models with prescribed 2/3-valuations including c₄ = 0 and c₆ = 0, and
//! u ∈ {2,3}-scaled non-minimal models): **0 mismatches** on the root
//! number, the Kodaira symbol AND the conductor exponent at both p = 2 and
//! p = 3, with **every row of both tables exercised** (44/44 rows of Table
//! III, 24/24 rows of Table II). No row of this port is
//! tested-by-transcription-only: the unit battery below re-checks one
//! PARI-verified exemplar per row (63 curves), and every additive call
//! cross-checks the fired row's (Kodaira, f) against this crate's own
//! Tate data at runtime (asserted in `crate::rootnumber`). The global root
//! numbers built from these tables are further validated against the
//! modular-symbols Fricke eigenvalues (ε = −w_N) in
//! `tests/modular_crosscheck.rs` — a derivation with no shared code at all.
//!
//! ## Scope
//!
//! The tables cover ALL reduction types at p ∈ {2, 3} (good and
//! multiplicative rows included), but [`crate::rootnumber`] consumes them
//! only for additive reduction; the good/multiplicative agreement with the
//! Tate-data-derived values is asserted in the tests here.

use crate::tate::KodairaSymbol;
use rustmath_integers::Integer;

/// One resolved table row: the local root number at p, plus the row's
/// Kodaira symbol and conductor exponent (used as live cross-checks
/// against Tate's algorithm) and the row index (for coverage tracking in
/// tests).
#[derive(Debug, Clone)]
pub(crate) struct RizzoLocal {
    /// The local root number W_p ∈ {±1}.
    pub sign: i8,
    /// The Kodaira type the fired row asserts (of the p-minimal model).
    pub kodaira: KodairaSymbol,
    /// The conductor exponent f_p the fired row asserts.
    pub conductor_exponent: u32,
    /// Row index within the table (Table II for p = 3, Table III for
    /// p = 2), in this port's fixed ordering.
    pub row: usize,
}

/// Sentinel for v_p(0) = ∞ (comfortably beyond any real valuation and
/// stable under the small subtractions below).
const INF: i64 = 1 << 40;

/// v_p(n) as i64 with the INF sentinel at n = 0.
fn vp(n: &Integer, p: i64) -> i64 {
    if n.is_zero() {
        INF
    } else {
        n.valuation(&Integer::from(p)) as i64
    }
}

/// The signed p-free part n / p^{v_p(n)} (0 for 0).
fn pfree(n: &Integer, p: i64) -> Integer {
    if n.is_zero() {
        return Integer::zero();
    }
    let k = n.valuation(&Integer::from(p));
    exact_div_pow(n, p, k)
}

/// n / p^k, asserting exactness (transcription-error detector).
fn exact_div_pow(n: &Integer, p: i64, k: u32) -> Integer {
    if n.is_zero() {
        return Integer::zero();
    }
    let d = Integer::from(p).pow(k);
    let q = n / &d;
    assert!(
        &q * &d == *n,
        "rizzo: non-exact division by {}^{} (bug)",
        p,
        k
    );
    q
}

/// n / p^k when exact, else 0. The cofactors c_{4,2}, c_{4,4}, c_{6,4},
/// c_{6,7} are consumed only on rows whose valuation pattern guarantees
/// divisibility, so the 0 fallback is never consumed (same contract as the
/// validated reference transcription).
fn div_pow_or_zero(n: &Integer, p: i64, k: u32) -> Integer {
    if n.is_zero() {
        return Integer::zero();
    }
    let d = Integer::from(p).pow(k);
    let q = n / &d;
    if &q * &d == *n {
        q
    } else {
        Integer::zero()
    }
}

/// The non-negative residue of x mod m, as i64 (m small).
fn md(x: &Integer, m: i64) -> i64 {
    x.modulo(&Integer::from(m)).to_i64()
}

/// The reduced triplet (a, b, c) and the descaled invariants
/// (c4r, c6r, Δr) after dividing the model by u = p^k.
#[allow(clippy::type_complexity)]
fn reduce(
    c4: &Integer,
    c6: &Integer,
    delta: &Integer,
    p: i64,
) -> (i64, i64, i64, Integer, Integer, Integer) {
    assert!(!delta.is_zero(), "rizzo: singular curve (Δ = 0)");
    let va = vp(c4, p);
    let vb = vp(c6, p);
    let vc = vp(delta, p);
    let k = (va / 4).min(vb / 6).min(vc / 12);
    assert!((0..INF / 24).contains(&k), "rizzo: k out of range (bug)");
    let c4r = exact_div_pow(c4, p, (4 * k) as u32);
    let c6r = exact_div_pow(c6, p, (6 * k) as u32);
    let dr = exact_div_pow(delta, p, (12 * k) as u32);
    let a = if va == INF { INF } else { va - 4 * k };
    let b = if vb == INF { INF } else { vb - 6 * k };
    (a, b, vc - 12 * k, c4r, c6r, dr)
}

/// A valuation pattern of a table row.
#[derive(Clone, Copy)]
enum Pat {
    Eq(i64),
    Ge(i64),
}

impl Pat {
    fn matches(self, v: i64) -> bool {
        match self {
            Pat::Eq(n) => v == n,
            Pat::Ge(n) => v >= n,
        }
    }
}

use Pat::{Eq, Ge};

/// One table row: valuation patterns, optional extra selector, Kodaira
/// type, conductor exponent, and whether W_p = +1 (the congruence side
/// conditions are all precomputed before the rows are built, so both
/// `selector` and `plus` are plain booleans here).
struct Row {
    a: Pat,
    b: Pat,
    c: Pat,
    selector: Option<bool>,
    kodaira: KodairaSymbol,
    f: u32,
    plus: bool,
}

/// Find the unique row matching (a, b, c) and resolve it. Exactly one row
/// must fire — 0 or ≥2 is a transcription bug and panics (never a guessed
/// answer).
fn fire(rows: Vec<Row>, a: i64, b: i64, c: i64, p: i64) -> RizzoLocal {
    let mut hit: Option<(usize, &Row)> = None;
    for (i, row) in rows.iter().enumerate() {
        if row.a.matches(a) && row.b.matches(b) && row.c.matches(c) && row.selector.unwrap_or(true)
        {
            assert!(
                hit.is_none(),
                "rizzo: rows {} and {} both fire for reduced triplet ({}, {}, {}) at p = {} (bug)",
                hit.unwrap().0,
                i,
                a,
                b,
                c,
                p
            );
            hit = Some((i, row));
        }
    }
    let (i, row) = hit.unwrap_or_else(|| {
        panic!(
            "rizzo: no row fires for reduced triplet ({}, {}, {}) at p = {} (bug)",
            a, b, c, p
        )
    });
    RizzoLocal {
        sign: if row.plus { 1 } else { -1 },
        kodaira: row.kodaira.clone(),
        conductor_exponent: row.f,
        row: i,
    }
}

/// W_3(E) via Rizzo Table II (all reduction types at p = 3), from the
/// c₄, c₆, Δ of ANY integral Weierstrass model of E.
pub(crate) fn rizzo_w3(c4: &Integer, c6: &Integer, delta: &Integer) -> RizzoLocal {
    let (a, b, c, c4r, c6r, dr) = reduce(c4, c6, delta, 3);
    let c4p = pfree(&c4r, 3); // c₄′
    let c6p = pfree(&c6r, 3); // c₆′
    let dp = pfree(&dr, 3); // Δ′
    let c42 = div_pow_or_zero(&c4r, 3, 2); // c_{4,2} (rows with a ≥ 2)
    let c44 = div_pow_or_zero(&c4r, 3, 4); // c_{4,4} (rows with a ≥ 4)

    // side conditions for the split rows (≥2,3,3) and (≥4,6,9)
    let special_a = md(
        &(&(&c6p * &c6p) + &(Integer::from(2) - Integer::from(3) * c42)),
        9,
    ) == 0;
    let special_b = md(
        &(&(&c6p * &c6p) + &(Integer::from(2) - Integer::from(3) * c44)),
        9,
    ) == 0;

    let c6p_m3 = md(&c6p, 3);
    let c6p_m9 = md(&c6p, 9);
    let c4p_m3 = md(&c4p, 3);
    let c4mc6_m3 = md(&(&c4p - &c6p), 3);
    let dmc6_m3 = md(&(&dp - &c6p), 3);

    let rows = vec![
        // 0: (0, 0, 0) — good reduction
        Row {
            a: Eq(0),
            b: Eq(0),
            c: Eq(0),
            selector: None,
            kodaira: KodairaSymbol::In(0),
            f: 0,
            plus: true,
        },
        // 1: (1, ≥3, 0) — good reduction
        Row {
            a: Eq(1),
            b: Ge(3),
            c: Eq(0),
            selector: None,
            kodaira: KodairaSymbol::In(0),
            f: 0,
            plus: true,
        },
        // 2: (0, 0, ≥1) — multiplicative I_c
        Row {
            a: Eq(0),
            b: Eq(0),
            c: Ge(1),
            selector: None,
            kodaira: KodairaSymbol::In(c.max(0) as u32),
            f: 1,
            plus: c6p_m3 == 1,
        },
        // 3: (1, 2, 0) — II*
        Row {
            a: Eq(1),
            b: Eq(2),
            c: Eq(0),
            selector: None,
            kodaira: KodairaSymbol::IIStar,
            f: 4,
            plus: true,
        },
        // 4: (≥2, 2, 1) — II*
        Row {
            a: Ge(2),
            b: Eq(2),
            c: Eq(1),
            selector: None,
            kodaira: KodairaSymbol::IIStar,
            f: 5,
            plus: c6p_m3 == 1,
        },
        // 5: (≥2, 3, 3), c₆′² + 2 ≢ 3c_{4,2} (9) — II
        Row {
            a: Ge(2),
            b: Eq(3),
            c: Eq(3),
            selector: Some(!special_a),
            kodaira: KodairaSymbol::II,
            f: 3,
            plus: c6p_m9 == 4 || c6p_m9 == 7 || c6p_m9 == 8,
        },
        // 6: (≥2, 3, 3), c₆′² + 2 ≡ 3c_{4,2} (9) — III
        Row {
            a: Ge(2),
            b: Eq(3),
            c: Eq(3),
            selector: Some(special_a),
            kodaira: KodairaSymbol::III,
            f: 2,
            plus: true,
        },
        // 7: (2, 4, 3) — II
        Row {
            a: Eq(2),
            b: Eq(4),
            c: Eq(3),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 3,
            plus: c4mc6_m3 != 0,
        },
        // 8: (2, ≥5, 3) — III
        Row {
            a: Eq(2),
            b: Ge(5),
            c: Eq(3),
            selector: None,
            kodaira: KodairaSymbol::III,
            f: 2,
            plus: true,
        },
        // 9: (2, 3, 4) — II
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Eq(4),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 4,
            plus: true,
        },
        // 10: (2, 3, 5) — IV
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Eq(5),
            selector: None,
            kodaira: KodairaSymbol::IV,
            f: 3,
            plus: dmc6_m3 == 0,
        },
        // 11: (≥3, 4, 5) — II
        Row {
            a: Ge(3),
            b: Eq(4),
            c: Eq(5),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 5,
            plus: c6p_m3 == 2,
        },
        // 12: (2, 3, ≥6) — I*_{c−6}
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Ge(6),
            selector: None,
            kodaira: KodairaSymbol::InStar((c - 6).max(0) as u32),
            f: 2,
            plus: false,
        },
        // 13: (3, 5, 6) — IV
        Row {
            a: Eq(3),
            b: Eq(5),
            c: Eq(6),
            selector: None,
            kodaira: KodairaSymbol::IV,
            f: 4,
            plus: c4p_m3 == 2,
        },
        // 14: (3, ≥6, 6) — I₀*
        Row {
            a: Eq(3),
            b: Ge(6),
            c: Eq(6),
            selector: None,
            kodaira: KodairaSymbol::InStar(0),
            f: 2,
            plus: false,
        },
        // 15: (≥4, 5, 7) — IV
        Row {
            a: Ge(4),
            b: Eq(5),
            c: Eq(7),
            selector: None,
            kodaira: KodairaSymbol::IV,
            f: 5,
            plus: c6p_m3 == 2,
        },
        // 16: (≥4, 6, 9), c₆′² + 2 ≡ 3c_{4,4} (9) — III*
        Row {
            a: Ge(4),
            b: Eq(6),
            c: Eq(9),
            selector: Some(special_b),
            kodaira: KodairaSymbol::IIIStar,
            f: 2,
            plus: true,
        },
        // 17: (4, 6, 9), c₆′² + 2 ≢ 3c_{4,4} (9) — IV*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(9),
            selector: Some(!special_b),
            kodaira: KodairaSymbol::IVStar,
            f: 3,
            plus: c6p_m9 == 4 || c6p_m9 == 8,
        },
        // 18: (≥5, 6, 9), c₆′² + 2 ≢ 3c_{4,4} (9) — IV* (corrected row:
        //     the printed side condition is negated; see module docs)
        Row {
            a: Ge(5),
            b: Eq(6),
            c: Eq(9),
            selector: Some(!special_b),
            kodaira: KodairaSymbol::IVStar,
            f: 3,
            plus: c6p_m9 == 1 || c6p_m9 == 2,
        },
        // 19: (4, 7, 9) — IV*
        Row {
            a: Eq(4),
            b: Eq(7),
            c: Eq(9),
            selector: None,
            kodaira: KodairaSymbol::IVStar,
            f: 3,
            plus: c6p_m3 == 2,
        },
        // 20: (4, ≥8, 9) — III*
        Row {
            a: Eq(4),
            b: Ge(8),
            c: Eq(9),
            selector: None,
            kodaira: KodairaSymbol::IIIStar,
            f: 2,
            plus: true,
        },
        // 21: (4, 6, 10) — IV*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(10),
            selector: None,
            kodaira: KodairaSymbol::IVStar,
            f: 4,
            plus: c6p_m9 == 2 || c6p_m9 == 7,
        },
        // 22: (4, 6, 11) — II*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(11),
            selector: None,
            kodaira: KodairaSymbol::IIStar,
            f: 3,
            plus: c6p_m3 == 1,
        },
        // 23: (≥5, 7, 11) — IV*
        Row {
            a: Ge(5),
            b: Eq(7),
            c: Eq(11),
            selector: None,
            kodaira: KodairaSymbol::IVStar,
            f: 5,
            plus: c6p_m3 == 1,
        },
    ];
    fire(rows, a, b, c, 3)
}

/// W_2(E) via Rizzo Table III (all reduction types at p = 2), from the
/// c₄, c₆, Δ of ANY integral Weierstrass model of E.
pub(crate) fn rizzo_w2(c4: &Integer, c6: &Integer, delta: &Integer) -> RizzoLocal {
    let (a, b, c, c4r, c6r, dr) = reduce(c4, c6, delta, 2);
    let c4p = pfree(&c4r, 2);
    let c6p = pfree(&c6r, 2);
    let dp = pfree(&dr, 2);
    let c64 = div_pow_or_zero(&c6r, 2, 4); // c_{6,4} (rows with b ≥ 5)
    let c67 = div_pow_or_zero(&c6r, 2, 7); // c_{6,7} (rows with b ≥ 7)

    let c4p_m4 = md(&c4p, 4);
    let c4p_m8 = md(&c4p, 8);
    let c6p_m4 = md(&c6p, 4);
    let c6p_m8 = md(&c6p, 8);
    let m4_c4mc6 = md(&(&c4p - &c6p), 4);
    let m4_dmc6 = md(&(&dp - &c6p), 4);
    let m4_d = md(&dp, 4);
    let m16_c4_4c6 = md(&(&c4p + &(Integer::from(4) * c6p.clone())), 16);
    let m16_c4_4c64 = md(&(&c4p + &(Integer::from(4) * c64.clone())), 16);
    let m16_c4m4c67 = md(&(&c4p - &(Integer::from(4) * c67.clone())), 16);
    let two_c6_c4 = &(Integer::from(2) * c6p.clone()) + &c4p;
    let m8_2c6_c4 = md(&two_c6_c4, 8);
    let m16_2c6_c4 = md(&two_c6_c4, 16);
    let m32_2c6_c4 = md(&two_c6_c4, 32);
    let m8_2c4_c6 = md(&(&(Integer::from(2) * c4p.clone()) + &c6p), 8);
    let m8_c4c6 = md(&(&c4p * &c6p), 8);
    let m4_c4c6 = md(&(&c4p * &c6p), 4);
    let m8_c6m5c4 = md(&(&c6p - &(Integer::from(5) * c4p.clone())), 8);
    let m64_c4m2c6 = md(&(&c4p - &(Integer::from(2) * c6p.clone())), 64);

    let rows = vec![
        // 0: (0, 0, 0), c₆′ ≡ 3 (4) — good reduction
        Row {
            a: Eq(0),
            b: Eq(0),
            c: Eq(0),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::In(0),
            f: 0,
            plus: true,
        },
        // 1: (0, 0, ≥0), c₆′ ≡ 1 (4) — I*_{c+4} (corrected row: the
        //    printed triplet (0,0,>0) misses c = 0; see module docs)
        Row {
            a: Eq(0),
            b: Eq(0),
            c: Ge(0),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::InStar((c + 4).max(0) as u32),
            f: 4,
            plus: false,
        },
        // 2: (3, 3, 0) — III*
        Row {
            a: Eq(3),
            b: Eq(3),
            c: Eq(0),
            selector: None,
            kodaira: KodairaSymbol::IIIStar,
            f: 5,
            plus: (c4p_m4 == 1 && (c6p_m8 == 1 || c6p_m8 == 7))
                || (c4p_m4 == 3 && (c6p_m8 == 1 || c6p_m8 == 3)),
        },
        // 3: (≥4, 3, 0), c₆′ ≡ 1 (4) — good reduction
        Row {
            a: Ge(4),
            b: Eq(3),
            c: Eq(0),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::In(0),
            f: 0,
            plus: true,
        },
        // 4: (≥4, 3, 0), c₆′ ≡ 3 (4) — II*
        Row {
            a: Ge(4),
            b: Eq(3),
            c: Eq(0),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::IIStar,
            f: 4,
            plus: false,
        },
        // 5: (2, ≥4, 0), c₄′ ≡ 3 (4) — I₂*
        Row {
            a: Eq(2),
            b: Ge(4),
            c: Eq(0),
            selector: Some(c4p_m4 == 3),
            kodaira: KodairaSymbol::InStar(2),
            f: 6,
            plus: b == 4,
        },
        // 6: (2, 4, 0), c₄′ ≡ 1 (4) — I₃*
        Row {
            a: Eq(2),
            b: Eq(4),
            c: Eq(0),
            selector: Some(c4p_m4 == 1),
            kodaira: KodairaSymbol::InStar(3),
            f: 5,
            plus: m16_c4_4c6 == 9 || m16_c4_4c6 == 13,
        },
        // 7: (2, ≥5, 0), c₄′ ≡ 1 (4) — I₃*
        Row {
            a: Eq(2),
            b: Ge(5),
            c: Eq(0),
            selector: Some(c4p_m4 == 1),
            kodaira: KodairaSymbol::InStar(3),
            f: 5,
            plus: m16_c4_4c64 == 5 || m16_c4_4c64 == 9,
        },
        // 8: (0, 0, ≥1), c₆′ ≡ 3 (4) — multiplicative I_c
        Row {
            a: Eq(0),
            b: Eq(0),
            c: Ge(1),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::In(c.max(0) as u32),
            f: 1,
            plus: c6p_m8 == 3,
        },
        // 9: (2, 3, 1) — I₂* (corrected row: the printed Kodaira symbol
        //    I₂ is a typo for I₂*; see module docs)
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Eq(1),
            selector: None,
            kodaira: KodairaSymbol::InStar(2),
            f: 7,
            plus: m16_c4_4c6 == 3 || md(&c4p, 16) == 11,
        },
        // 10: (2, 3, 2) — I₄*
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Eq(2),
            selector: None,
            kodaira: KodairaSymbol::InStar(4),
            f: 6,
            plus: m4_dmc6 == 0,
        },
        // 11: (3, 4, 2) — III*
        Row {
            a: Eq(3),
            b: Eq(4),
            c: Eq(2),
            selector: None,
            kodaira: KodairaSymbol::IIIStar,
            f: 7,
            plus: (c4p_m8 == 1 && (c6p_m8 == 5 || c6p_m8 == 7))
                || (c4p_m8 == 3 && (c6p_m8 == 3 || c6p_m8 == 5))
                || (c4p_m8 == 5 && (c6p_m8 == 1 || c6p_m8 == 3))
                || (c4p_m8 == 7 && (c6p_m8 == 1 || c6p_m8 == 7)),
        },
        // 12: (≥4, 4, 2) — II*
        Row {
            a: Ge(4),
            b: Eq(4),
            c: Eq(2),
            selector: None,
            kodaira: KodairaSymbol::IIStar,
            f: 6,
            plus: c6p_m4 == 1,
        },
        // 13: (2, 3, 3) — I₅*
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Eq(3),
            selector: None,
            kodaira: KodairaSymbol::InStar(5),
            f: 6,
            plus: m4_d == 3,
        },
        // 14: (3, 5, 3) — III*
        Row {
            a: Eq(3),
            b: Eq(5),
            c: Eq(3),
            selector: None,
            kodaira: KodairaSymbol::IIIStar,
            f: 8,
            plus: m8_2c6_c4 == 1 || m8_2c6_c4 == 3,
        },
        // 15: (3, ≥6, 3) — III*
        Row {
            a: Eq(3),
            b: Ge(6),
            c: Eq(3),
            selector: None,
            kodaira: KodairaSymbol::IIIStar,
            f: 8,
            plus: c4p_m8 == 5 || c4p_m8 == 7,
        },
        // 16: (2, 3, ≥4) — I*_{c+2}
        Row {
            a: Eq(2),
            b: Eq(3),
            c: Ge(4),
            selector: None,
            kodaira: KodairaSymbol::InStar((c + 2).max(0) as u32),
            f: 6,
            plus: c6p_m4 == 3,
        },
        // 17: (4, 5, 4), c₄′ ≡ c₆′ (4) — II
        Row {
            a: Eq(4),
            b: Eq(5),
            c: Eq(4),
            selector: Some(m4_c4mc6 == 0),
            kodaira: KodairaSymbol::II,
            f: 4,
            plus: c4p_m4 == 1,
        },
        // 18: (4, 5, 4), c₄′ ≡ 1, c₆′ ≡ 3 (4) — III
        Row {
            a: Eq(4),
            b: Eq(5),
            c: Eq(4),
            selector: Some(c4p_m4 == 1 && c6p_m4 == 3),
            kodaira: KodairaSymbol::III,
            f: 3,
            plus: m8_c4c6 == 3,
        },
        // 19: (4, 5, 4), c₄′ ≡ 3, c₆′ ≡ 1 (4) — IV
        Row {
            a: Eq(4),
            b: Eq(5),
            c: Eq(4),
            selector: Some(c6p_m4 == 1 && c4p_m4 == 3),
            kodaira: KodairaSymbol::IV,
            f: 2,
            plus: false,
        },
        // 20: (≥5, 5, 4), c₆′ ≡ 3 (4) — II
        Row {
            a: Ge(5),
            b: Eq(5),
            c: Eq(4),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::II,
            f: 4,
            plus: a == 5,
        },
        // 21: (5, 5, 4), c₆′ ≡ 1 (4) — III
        Row {
            a: Eq(5),
            b: Eq(5),
            c: Eq(4),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::III,
            f: 3,
            plus: c6p_m8 == 5,
        },
        // 22: (≥6, 5, 4), c₆′ ≡ 1 (4) — IV
        Row {
            a: Ge(6),
            b: Eq(5),
            c: Eq(4),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::IV,
            f: 2,
            plus: false,
        },
        // 23: (5, 6, 6) — II
        Row {
            a: Eq(5),
            b: Eq(6),
            c: Eq(6),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 6,
            plus: c4p_m4 == 3,
        },
        // 24: (≥6, 6, 6) — II
        Row {
            a: Ge(6),
            b: Eq(6),
            c: Eq(6),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 6,
            plus: c6p_m4 == 1,
        },
        // 25: (4, ≥7, 6), c₄′ ≡ 1 (4) — II
        Row {
            a: Eq(4),
            b: Ge(7),
            c: Eq(6),
            selector: Some(c4p_m4 == 1),
            kodaira: KodairaSymbol::II,
            f: 6,
            plus: b == 7,
        },
        // 26: (4, ≥7, 6), c₄′ ≡ 3 (4) — III
        Row {
            a: Eq(4),
            b: Ge(7),
            c: Eq(6),
            selector: Some(c4p_m4 == 3),
            kodaira: KodairaSymbol::III,
            f: 5,
            plus: m16_c4m4c67 == 7 || m16_c4m4c67 == 11,
        },
        // 27: (4, 6, 7) — II
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(7),
            selector: None,
            kodaira: KodairaSymbol::II,
            f: 7,
            plus: c6p_m8 == 5 || m8_c6m5c4 == 0,
        },
        // 28: (4, 6, 8), 2c₆′ + c₄′ ≡ 3 or 15 (16) — I₀*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(8),
            selector: Some(m16_2c6_c4 == 3 || m16_2c6_c4 == 15),
            kodaira: KodairaSymbol::InStar(0),
            f: 4,
            plus: m16_2c6_c4 == 3,
        },
        // 29: (4, 6, 8), 2c₆′ + c₄′ ≡ 7 (16) — I₁*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(8),
            selector: Some(m16_2c6_c4 == 7),
            kodaira: KodairaSymbol::InStar(1),
            f: 3,
            plus: m32_2c6_c4 == 23,
        },
        // 30: (4, 6, 8), 2c₆′ + c₄′ ≡ 11 (16) — IV*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(8),
            selector: Some(m16_2c6_c4 == 11),
            kodaira: KodairaSymbol::IVStar,
            f: 2,
            plus: false,
        },
        // 31: (5, 7, 8) — III
        Row {
            a: Eq(5),
            b: Eq(7),
            c: Eq(8),
            selector: None,
            kodaira: KodairaSymbol::III,
            f: 7,
            plus: m8_2c4_c6 == 7 || c6p_m8 == 3,
        },
        // 32: (≥6, 7, 8), c₆′ ≡ 3 (4) — I₀*
        Row {
            a: Ge(6),
            b: Eq(7),
            c: Eq(8),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::InStar(0),
            f: 4,
            plus: a == 6,
        },
        // 33: (6, 7, 8), c₆′ ≡ 1 (4) — I₁*
        Row {
            a: Eq(6),
            b: Eq(7),
            c: Eq(8),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::InStar(1),
            f: 3,
            plus: m8_2c4_c6 == 3,
        },
        // 34: (≥7, 7, 8), c₆′ ≡ 1 (4) — IV*
        Row {
            a: Ge(7),
            b: Eq(7),
            c: Eq(8),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::IVStar,
            f: 2,
            plus: false,
        },
        // 35: (4, 6, 9) — I₀*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(9),
            selector: None,
            kodaira: KodairaSymbol::InStar(0),
            f: 5,
            plus: m32_2c6_c4 == 11 || c6p_m8 == 7,
        },
        // 36: (5, 8, 9) — III
        Row {
            a: Eq(5),
            b: Eq(8),
            c: Eq(9),
            selector: None,
            kodaira: KodairaSymbol::III,
            f: 8,
            plus: m8_2c6_c4 == 1 || m8_2c6_c4 == 7,
        },
        // 37: (5, ≥9, 9) — III
        Row {
            a: Eq(5),
            b: Ge(9),
            c: Eq(9),
            selector: None,
            kodaira: KodairaSymbol::III,
            f: 8,
            plus: c4p_m8 == 1 || c4p_m8 == 3,
        },
        // 38: (4, 6, 10), c₆′ ≡ 1 (4) — I₂*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(10),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::InStar(2),
            f: 4,
            plus: true,
        },
        // 39: (4, 6, 10), c₆′ ≡ 3 (4) — III*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(10),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::IIIStar,
            f: 3,
            plus: m64_c4m2c6 == 3 || m64_c4m2c6 == 19,
        },
        // 40: (6, 8, 10) — I₀*
        Row {
            a: Eq(6),
            b: Eq(8),
            c: Eq(10),
            selector: None,
            kodaira: KodairaSymbol::InStar(0),
            f: 6,
            plus: m4_c4c6 == 3,
        },
        // 41: (≥7, 8, 10) — I₀*
        Row {
            a: Ge(7),
            b: Eq(8),
            c: Eq(10),
            selector: None,
            kodaira: KodairaSymbol::InStar(0),
            f: 6,
            plus: c6p_m4 == 1,
        },
        // 42: (4, 6, 11), c₆′ ≡ 1 (4) — I₃*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(11),
            selector: Some(c6p_m4 == 1),
            kodaira: KodairaSymbol::InStar(3),
            f: 4,
            plus: true,
        },
        // 43: (4, 6, 11), c₆′ ≡ 3 (4) — II*
        Row {
            a: Eq(4),
            b: Eq(6),
            c: Eq(11),
            selector: Some(c6p_m4 == 3),
            kodaira: KodairaSymbol::IIStar,
            f: 3,
            plus: c6p_m8 == 3,
        },
    ];
    fire(rows, a, b, c, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::EllipticCurve;
    use crate::tate::ReductionType;
    use std::collections::BTreeSet;

    fn curve(a: [i64; 5]) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a[0]),
            Integer::from(a[1]),
            Integer::from(a[2]),
            Integer::from(a[3]),
            Integer::from(a[4]),
        )
    }

    /// One PARI-verified exemplar per table row (63 curves hitting all 44
    /// rows of Table III and all 24 rows of Table II). The expected
    /// (W₂, W₃) pairs are PARI `ellrootno` outputs, checked against the
    /// reference transcription at emission time; the (Kodaira, f)
    /// cross-check against this crate's Tate data runs inside the table
    /// lookup assertions below.
    const EXEMPLARS: [([i64; 5], i8, i8); 63] = [
        ([0, -1, 0, -13, -11], -1, 1),
        ([0, -1, 0, -2, -2], 1, 1),
        ([0, -1, 0, -1, 5], 1, 1),
        ([0, -1, 0, 0, 4], -1, 1),
        ([0, -1, 0, 1, -1], -1, 1),
        ([0, -1, 0, 2, -3], -1, 1),
        ([0, -1, 0, 2, -1], -1, 1),
        ([0, -1, 0, 51, -43], 1, 1),
        ([0, -1, 1, 0, 0], 1, 1),
        ([0, 0, 0, -108, -162], -1, -1),
        ([0, 0, 0, -54, -27], -1, -1),
        ([0, 0, 0, -44, -16], 1, 1),
        ([0, 0, 0, -27, 0], -1, 1),
        ([0, 0, 0, -11, 10], 1, 1),
        ([0, 0, 0, -4, -4], 1, 1),
        ([0, 0, 0, -4, 0], -1, 1),
        ([0, 0, 0, -3, -1], 1, 1),
        ([0, 0, 0, -3, 0], -1, 1),
        ([0, 0, 0, -2, -1], 1, 1),
        ([0, 0, 0, -1, 0], -1, 1),
        ([0, 0, 0, 0, -16], -1, 1),
        ([0, 0, 0, 0, 1], -1, 1),
        ([0, 0, 0, 0, 2], 1, 1),
        ([0, 0, 0, 0, 3], -1, -1),
        ([0, 0, 0, 0, 8], 1, 1),
        ([0, 0, 0, 0, 27], -1, 1),
        ([0, 0, 0, 0, 81], -1, 1),
        ([0, 0, 0, 0, 243], -1, 1),
        ([0, 0, 0, 1, -6], -1, 1),
        ([0, 0, 0, 1, 6], -1, 1),
        ([0, 0, 0, 2, -8], -1, 1),
        ([0, 0, 0, 2, 0], -1, 1),
        ([0, 0, 0, 2, 2], -1, 1),
        ([0, 0, 0, 3, -3], -1, -1),
        ([0, 0, 0, 4, 0], -1, 1),
        ([0, 0, 0, 8, -16], -1, 1),
        ([0, 0, 0, 8, 0], 1, 1),
        ([0, 0, 0, 9, 0], -1, -1),
        ([0, 0, 0, 12, -16], 1, 1),
        ([0, 0, 0, 13, -14], -1, 1),
        ([0, 0, 0, 13, -2], -1, 1),
        ([0, 0, 0, 36, 32], 1, -1),
        ([0, 0, 0, 54, 54], 1, -1),
        ([0, 0, 0, 324, 243], -1, 1),
        ([0, 0, 1, -27, -34], 1, -1),
        ([0, 1, 0, -16, -12], 1, 1),
        ([0, 1, 0, -4, 0], -1, 1),
        ([0, 1, 0, -3, -2], -1, 1),
        ([0, 1, 0, -1, 7], 1, 1),
        ([0, 1, 0, 3, -1], -1, 1),
        ([0, 1, 0, 11, 19], -1, 1),
        ([0, 1, 0, 15, 15], -1, -1),
        ([0, 1, 0, 24, -16], 1, 1),
        ([0, 1, 0, 31, -1], 1, 1),
        ([0, 1, 1, 1, -1], 1, -1),
        ([1, -1, 0, -15, 35], 1, -1),
        ([1, -1, 0, -1, 0], 1, 1),
        ([1, -1, 0, 0, 3], 1, 1),
        ([1, -1, 0, 0, 4], 1, 1),
        ([1, -1, 0, 3, 8], 1, -1),
        ([1, -1, 1, -2, -8], 1, 1),
        ([1, -1, 1, 4, -7], -1, -1),
        ([1, 0, 1, 0, 0], 1, 1),
    ];

    /// THE TRANSCRIPTION GATE: every exemplar reproduces PARI's W₂ and W₃,
    /// the fired row's (Kodaira, f) matches this crate's Tate data at both
    /// primes, all 68 rows of the two tables are covered, and at good /
    /// multiplicative reduction the table sign agrees with the Tate-data
    /// derivation of [`crate::rootnumber`].
    #[test]
    fn test_rizzo_exemplar_battery_covers_every_row() {
        let mut rows2 = BTreeSet::new();
        let mut rows3 = BTreeSet::new();
        for (a, w2_expect, w3_expect) in &EXEMPLARS {
            let e = curve(*a);
            let (c4, c6) = e.c_invariants();
            let r2 = rizzo_w2(&c4, &c6, &e.discriminant);
            let r3 = rizzo_w3(&c4, &c6, &e.discriminant);
            assert_eq!(r2.sign, *w2_expect, "W_2 of {:?} (PARI ellrootno)", a);
            assert_eq!(r3.sign, *w3_expect, "W_3 of {:?} (PARI ellrootno)", a);
            rows2.insert(r2.row);
            rows3.insert(r3.row);
            for (p, r) in [(2i64, &r2), (3i64, &r3)] {
                let ld = e.local_data(&Integer::from(p));
                assert_eq!(
                    r.kodaira, ld.kodaira,
                    "Kodaira at {} of {:?}: table row vs Tate",
                    p, a
                );
                assert_eq!(
                    r.conductor_exponent, ld.conductor_exponent,
                    "f_{} of {:?}: table row vs Tate",
                    p, a
                );
                // non-additive reduction: the table must agree with the
                // independent Tate-data-derived local root number
                let derived = match ld.reduction {
                    ReductionType::Good => Some(1i8),
                    ReductionType::SplitMultiplicative => Some(-1),
                    ReductionType::NonsplitMultiplicative => Some(1),
                    ReductionType::Additive => None,
                };
                if let Some(d) = derived {
                    assert_eq!(r.sign, d, "table vs Tate derivation at {} of {:?}", p, a);
                }
            }
        }
        assert_eq!(rows2.len(), 44, "all 44 rows of Table III exercised");
        assert_eq!(rows3.len(), 24, "all 24 rows of Table II exercised");
        assert_eq!(*rows2.iter().max().unwrap(), 43);
        assert_eq!(*rows3.iter().max().unwrap(), 23);
    }

    /// The k-reduction makes the tables model-independent: u-scaled
    /// (u = 2, 3, 6) non-minimal models give identical answers.
    #[test]
    fn test_rizzo_scaling_invariance() {
        for (a, _, _) in EXEMPLARS.iter().take(20) {
            let e = curve(*a);
            let (c4, c6) = e.c_invariants();
            let base2 = rizzo_w2(&c4, &c6, &e.discriminant);
            let base3 = rizzo_w3(&c4, &c6, &e.discriminant);
            for u in [2i64, 3, 6] {
                let uu = Integer::from(u);
                let c4s = &c4 * &uu.pow(4);
                let c6s = &c6 * &uu.pow(6);
                let ds = &e.discriminant * &uu.pow(12);
                let s2 = rizzo_w2(&c4s, &c6s, &ds);
                let s3 = rizzo_w3(&c4s, &c6s, &ds);
                assert_eq!(s2.sign, base2.sign, "W_2 of {:?} scaled by {}", a, u);
                assert_eq!(s3.sign, base3.sign, "W_3 of {:?} scaled by {}", a, u);
                assert_eq!(s2.kodaira, base2.kodaira);
                assert_eq!(s3.kodaira, base3.kodaira);
            }
        }
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn test_rizzo_rejects_singular() {
        let _ = rizzo_w2(&Integer::zero(), &Integer::zero(), &Integer::zero());
    }
}
