//! # Local and global root numbers of elliptic curves over Q
//!
//! The global root number w(E) ∈ {±1} is the sign of the functional
//! equation of the (analytically continued) L-function:
//! Λ(s) = w(E)·Λ(2−s), where Λ(s) = N^{s/2}(2π)^{−s}Γ(s)L(E,s). It factors
//! as a product of local root numbers over all places,
//!
//! ```text
//!     w(E) = w_∞ · ∏_p w_p(E),      w_p(E) = +1 at good primes,
//! ```
//!
//! so the product is finite (bad primes only). Every local factor
//! implemented here is derived from the EXISTING Tate local data
//! ([`crate::tate`]); nothing is guessed:
//!
//! * **Archimedean place**: w_∞ = −1 for every elliptic curve over R
//!   (Rohrlich, *Galois theory, elliptic curves, and root numbers*,
//!   Compositio 100 (1996), Prop. at §2: the (2-dimensional) archimedean
//!   representation attached to E/R has root number i^2 = −1, matching the
//!   weight-2 modular normalization).
//!
//! * **Multiplicative reduction** (Kodaira I_n, n ≥ 1): the Weil–Deligne
//!   representation is the special (Steinberg) representation sp(2)
//!   twisted by the unramified character χ with χ(Frob_p) = +1 (split,
//!   Tate curve) or −1 (non-split quadratic unramified twist). Its root
//!   number is −χ(Frob_p) (Rohrlich 1996, Prop. 3(ii); Tate-curve
//!   computation), i.e.
//!   **w_p = −1 iff split multiplicative, +1 iff non-split**.
//!   The split test itself is Tate step 3 (tangent directions at the node
//!   rational over F_p), PARI `ellap`-validated in [`crate::tate`]'s tests.
//!
//! * **Additive reduction, p ≥ 5** (tame case; Rohrlich, *Variation of the
//!   root number in families of elliptic curves*, Compositio 87 (1993),
//!   Prop. 2; same table as PARI's `ellrootno` and the Dokchitser–
//!   Dokchitser parity survey):
//!   - potentially good reduction with e = 12/gcd(v_p(Δ_min), 12):
//!     * e = 2 or 6 (Kodaira II, II*, I0*):  w_p = (−1/p)
//!     * e = 4      (Kodaira III, III*):     w_p = (−2/p)
//!     * e = 3      (Kodaira IV, IV*):       w_p = (−3/p)
//!   - potentially multiplicative reduction (Kodaira I_n*, n ≥ 1): E is
//!     the quadratic twist of a Tate curve by the ramified character
//!     χ_d with p | d, and w_p = χ_d(−1) = (−1, d)_p = (−1/p) — the same
//!     value as the e ∈ {2,6} row, so ALL I_m* types (m ≥ 0) use (−1/p).
//!
//!   In Kodaira terms (the form implemented below, independent of any
//!   model choice):
//!   **II, II*, I_m* (m ≥ 0) → (−1/p);  III, III* → (−2/p);
//!   IV, IV* → (−3/p)**.
//!
//! * **Additive reduction at p = 2 or 3** (wild ramification): the
//!   complete case tables of Kraus (p = 3) and Halberstadt (p = 2),
//!   Comptes Rendus 326 (1998), in Rizzo's explicit tabulation, keyed on
//!   (v(c4), v(c6), v(Δ)) plus congruence side conditions — implemented in
//!   the crate-private `rizzo` module (see there for the transcription provenance, the
//!   applied published corrections, and the 51,212-curve PARI validation
//!   with full row coverage). Every additive call cross-checks the fired
//!   row's (Kodaira type, conductor exponent) against this crate's own
//!   Tate data — a live transcription-error detector. The `Err` arm of
//!   [`local_root_number`] is therefore no longer produced for any
//!   nonsingular curve; the `Result` signature is kept for API stability.
//!
//! ## Independent validation (performed BEFORE these tests were written)
//!
//! The table above was validated against a completely independent
//! derivation: for 20+ curves (every Kodaira type at p ≥ 5, both values of
//! each quadratic character class, plus split/non-split multiplicative
//! mixes at 2, 3 and larger primes) the functional-equation sign ε = w(E)
//! was pinned numerically in Python by the split-point independence of
//! Λ(1) computed from point-counted a_n at two different split points
//! (agreement to 12+ digits for the true sign, gross failure for the
//! wrong one). All instances matched the local product. The
//! multiplicative-prime rule is additionally cross-checked against the
//! chunk-5 Fricke eigenvalues of the corresponding newforms (Eichler–
//! Shimura) in `tests/modular_crosscheck.rs` — and so are the wild
//! p ∈ {2, 3} additive factors, on a battery of additive-conductor levels
//! (20, 24, 27, 32, 36, …), closing the loop the stage-2 spec demands:
//! w(E) from Tate + Rizzo tables ≡ −w_N from the exact rational Fricke
//! matrix of the attached newform.

use crate::curve::EllipticCurve;
use crate::tate::{KodairaSymbol, ReductionType};
use rustmath_integers::prime::{factor, is_prime};
use rustmath_integers::Integer;

/// The Legendre symbol (a/p) for a small nonzero integer a and an odd prime
/// p ≥ 5 not dividing a, via [`Integer::legendre_symbol`] on the reduced
/// (non-negative) residue — the reduction first matters, see the note in
/// [`crate::tate`] about negative inputs.
fn legendre_of(a: i64, p: &Integer) -> i8 {
    let r = Integer::from(a).modulo(p);
    assert!(
        !r.is_zero(),
        "legendre_of: p = {} divides a = {} (caller must ensure p >= 5)",
        p,
        a
    );
    let s = r
        .legendre_symbol(p)
        .expect("p is an odd prime by construction");
    assert!(s == 1 || s == -1, "nonzero residue has symbol ±1");
    s
}

/// The local root number w_p(E) at the prime p, from Tate local data
/// (p ≥ 5 additive: Rohrlich; p ∈ {2, 3} additive: the Kraus/Halberstadt
/// tables via the crate-private `rizzo` module).
///
/// The `Err` arm is no longer produced for any nonsingular curve (the
/// wild tables are implemented); the `Result` signature is kept for API
/// stability.
///
/// # Panics
///
/// Panics if p is not prime or the curve is singular — and, as a live
/// wrongness detector, if the Rizzo-table row fired at a wild additive
/// prime disagrees with Tate's algorithm about the Kodaira type or the
/// conductor exponent (a transcription bug, never an answer).
pub fn local_root_number(curve: &EllipticCurve, p: &Integer) -> Result<i8, String> {
    assert!(is_prime(p), "local_root_number: p = {} is not prime", p);
    let ld = curve.local_data(p);
    match ld.reduction {
        ReductionType::Good => Ok(1),
        ReductionType::SplitMultiplicative => Ok(-1),
        ReductionType::NonsplitMultiplicative => Ok(1),
        ReductionType::Additive => {
            if *p == Integer::from(2) || *p == Integer::from(3) {
                let (c4, c6) = curve.c_invariants();
                let r = if *p == Integer::from(2) {
                    crate::rizzo::rizzo_w2(&c4, &c6, &curve.discriminant)
                } else {
                    crate::rizzo::rizzo_w3(&c4, &c6, &curve.discriminant)
                };
                // Live transcription-error detector: the fired row's
                // Kodaira type and conductor exponent must agree with
                // Tate's algorithm (independent derivations).
                assert_eq!(
                    r.kodaira, ld.kodaira,
                    "rizzo row {} at p = {}: Kodaira {} vs Tate {} (transcription bug)",
                    r.row, p, r.kodaira, ld.kodaira
                );
                assert_eq!(
                    r.conductor_exponent, ld.conductor_exponent,
                    "rizzo row {} at p = {}: f = {} vs Tate {} (transcription bug)",
                    r.row, p, r.conductor_exponent, ld.conductor_exponent
                );
                return Ok(r.sign);
            }
            let a = match ld.kodaira {
                KodairaSymbol::II | KodairaSymbol::IIStar | KodairaSymbol::InStar(_) => -1i64,
                KodairaSymbol::III | KodairaSymbol::IIIStar => -2,
                KodairaSymbol::IV | KodairaSymbol::IVStar => -3,
                KodairaSymbol::In(_) => {
                    unreachable!("Kodaira I_n is not additive: Tate bug")
                }
            };
            Ok(legendre_of(a, p))
        }
    }
}

/// The global root number w(E) = w_∞ · ∏_{p | Δ} w_p(E) (good primes,
/// including primes where the given model is non-minimal, contribute +1).
/// This is the sign ε of the functional equation Λ(s) = ε·Λ(2−s) of the
/// analytically continued L-function — the analytic continuation itself
/// being the modularity theorem (Wiles, Taylor–Wiles, BCDT), with the
/// conductor/ε-factor match due to Carayol.
///
/// Returns `Err` when any local factor is unresolved (additive reduction
/// at 2 or 3; see [`local_root_number`]).
///
/// # Panics
///
/// Panics if the curve is singular.
pub fn global_root_number(curve: &EllipticCurve) -> Result<i8, String> {
    assert!(
        !curve.is_singular(),
        "global_root_number: curve is singular"
    );
    let mut w: i8 = -1; // the archimedean place
    for (p, _) in factor(&curve.discriminant.abs()) {
        w *= local_root_number(curve, &p)?;
    }
    Ok(w)
}

impl EllipticCurve {
    /// The local root number w_p(E) at p; see
    /// [`local_root_number`](crate::rootnumber::local_root_number).
    pub fn local_root_number(&self, p: &Integer) -> Result<i8, String> {
        local_root_number(self, p)
    }

    /// The global root number w(E) (the functional-equation sign ε); see
    /// [`global_root_number`](crate::rootnumber::global_root_number).
    /// `Err` = honest refusal (additive reduction at 2 or 3), never a
    /// guess.
    pub fn root_number(&self) -> Result<i8, String> {
        global_root_number(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a1),
            Integer::from(a2),
            Integer::from(a3),
            Integer::from(a4),
            Integer::from(a6),
        )
    }

    /// Multiplicative-only curves: w(E) must equal the functional-equation
    /// sign pinned independently in Python (split-point independence of
    /// Λ(1) from point-counted a_n; every value below was derived BEFORE
    /// this test). The same signs are cross-checked against Fricke
    /// eigenvalues in tests/modular_crosscheck.rs.
    #[test]
    fn test_global_root_number_multiplicative_battery() {
        let cases: [(&str, [i64; 5], i8); 12] = [
            ("11a1", [0, -1, 1, -10, -20], 1),
            ("14a1", [1, 0, 1, 4, -6], 1),
            ("15a1", [1, 1, 1, -10, -10], 1),
            ("17a1", [1, -1, 1, -1, -14], 1),
            ("19a1", [0, 1, 1, -9, -15], 1),
            ("21a1", [1, 0, 0, -4, -1], 1),
            ("26a1", [1, 0, 1, -5, -8], 1),
            ("26b1", [1, -1, 1, -3, 3], 1),
            ("37a1", [0, 0, 1, -1, 0], -1),
            ("37b1", [0, 1, 1, -23, -50], 1),
            ("65a1", [1, 0, 0, -1, 0], -1),
            ("389a1", [0, 1, 1, -2, 0], 1),
        ];
        for (label, a, eps) in &cases {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            assert_eq!(e.root_number(), Ok(*eps), "global root number of {}", label);
        }
    }

    /// The additive battery, p ≥ 5: every row of the Rohrlich table in
    /// both quadratic-character classes. Each model was found by a search
    /// constrained to be globally minimal, additive ONLY at the target
    /// prime with the target Kodaira type, and multiplicative at every
    /// other bad prime; its global sign was then pinned by the independent
    /// Python split-point test (all PASS, derived before this test).
    #[test]
    fn test_global_root_number_additive_battery() {
        // (kodaira@p, [a1,a2,a3,a4,a6], pinned epsilon)
        let cases: [(&str, [i64; 5], i8); 13] = [
            ("II@5", [1, 1, 0, -45, -185], -1),
            ("II@7", [1, 5, 0, -182, 532], 1),
            ("III@5", [1, 1, 0, -80, 0], -1),
            ("III@7(49a1)", [1, -1, 0, -2, -1], 1),
            ("III@11", [1, 8, 0, -88, -726], 1),
            ("IV@5", [1, 1, 0, -75, 225], 1),
            ("IV@7", [1, 5, 0, -196, 784], -1),
            ("I0*@5", [1, 1, 0, -200, 0], 1),
            ("I1*@5", [1, 1, 0, -250, -1625], -1),
            ("I1*@7", [1, 5, 0, -343, 1029], -1),
            ("IV*@5", [1, 1, 0, -75, -1625], 1),
            ("IV*@7", [1, 5, 0, 245, 343], 1),
            ("II*@5", [1, 1, 0, 300, -1000], -1),
        ];
        for (label, a, eps) in &cases {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            assert_eq!(e.root_number(), Ok(*eps), "root number, case {}", label);
        }
        // III*@5, pinned +1 by the split test (its only other bad prime is
        // 61, where a_61 = −1: non-split I1): also pin the local factors
        // individually — (−1)·(−1)·(+1) = +1.
        let e = curve(1, 1, 0, -200, 875);
        assert_eq!(
            e.local_root_number(&Integer::from(5)),
            Ok(-1),
            "III* at 5: (-2/5) = -1"
        );
        assert_eq!(e.compute_a_p(&Integer::from(61)), -1, "non-split at 61");
        assert_eq!(
            e.local_root_number(&Integer::from(61)),
            Ok(1),
            "non-split I1 at 61"
        );
        assert_eq!(e.root_number(), Ok(1), "III*@5 global");
    }

    /// I0* at 7 via the quadratic twist of 11a1 by −7 (a non-minimal
    /// large-coefficient model; Tate minimalizes internally). Pinned
    /// ε = −1 by the Python split test with N = 539.
    #[test]
    fn test_global_root_number_twist_i0star_at_7() {
        let e = curve(0, 0, 0, -656208, 370588176);
        // sanity: conductor 539 = 7^2 * 11 (I0* at 7, multiplicative at 11)
        assert_eq!(e.compute_conductor(), Integer::from(539));
        let ld7 = e.local_data(&Integer::from(7));
        assert_eq!(ld7.kodaira.to_string(), "I0*");
        assert_eq!(
            e.local_root_number(&Integer::from(7)),
            Ok(-1),
            "(-1/7) = -1"
        );
        assert_eq!(e.root_number(), Ok(-1));
    }

    /// The former honest refusals at wild additive primes are now DECIDED
    /// by the Kraus/Halberstadt tables (see the crate-private `rizzo` module): every local
    /// and global sign below was derived from PARI `ellrootno` BEFORE this
    /// test (and the same signs are validated against modular Fricke
    /// eigenvalues in tests/modular_crosscheck.rs).
    #[test]
    fn test_root_number_wild_additive_decided() {
        // y² = x³ − x: N = 32, Kodaira III at 2, w₂ = −1, w = +1.
        let e = curve(0, 0, 0, -1, 0);
        assert_eq!(e.local_root_number(&Integer::from(2)), Ok(-1));
        assert_eq!(e.root_number(), Ok(1));
        // y² = x³ + 1: N = 36, additive at BOTH 2 (IV, w₂ = −1) and
        // 3 (III, w₃ = +1); w = (−1)·(−1)·(+1) = +1.
        let e = curve(0, 0, 0, 0, 1);
        assert_eq!(e.local_root_number(&Integer::from(2)), Ok(-1));
        assert_eq!(e.local_root_number(&Integer::from(3)), Ok(1));
        assert_eq!(e.root_number(), Ok(1));
        // 27a1 (y² + y = x³ − 7): N = 27, IV* at 3, w₃ = −1, w = +1.
        let e = curve(0, 0, 1, 0, -7);
        assert_eq!(e.local_root_number(&Integer::from(3)), Ok(-1));
        assert_eq!(e.root_number(), Ok(1));
        // the local factor at a GOOD prime of the same curve stays +1
        assert_eq!(e.local_root_number(&Integer::from(5)), Ok(1));
    }

    /// GLOBAL wild battery: additive-at-2/3 conductors spanning 20..126,
    /// both global signs, and models additive at 2, at 3, and at both.
    /// Every expected sign is a PARI `ellrootno` output derived before
    /// this test was written.
    #[test]
    fn test_global_root_number_wild_battery() {
        let cases: [([i64; 5], i64, i8); 16] = [
            ([0, 1, 0, -1, 0], 20, 1),
            ([0, -1, 0, -4, 4], 24, 1),
            ([0, 0, 1, 0, 0], 27, 1),
            ([0, 0, 0, 4, 0], 32, 1),
            ([0, 0, 0, 0, -1], 144, 1),
            ([0, 0, 0, -7, -6], 40, 1),
            ([0, 1, 0, 3, -1], 44, 1),
            ([1, -1, 0, 0, -5], 45, 1),
            ([0, 1, 0, 1, 0], 48, 1),
            ([1, -1, 1, 1, -1], 54, 1),
            ([0, 0, 0, 1, 2], 56, 1),
            ([1, -1, 0, 9, 0], 63, 1),
            ([0, 0, 0, 6, -7], 72, 1),
            ([0, 0, 0, -4, 4], 88, -1),
            ([1, -1, 1, -2, 0], 99, -1),
            ([0, 1, 0, 0, 4], 112, -1),
        ];
        for (a, n, w) in &cases {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            assert_eq!(
                e.compute_conductor(),
                Integer::from(*n),
                "conductor of {:?}",
                a
            );
            assert_eq!(e.root_number(), Ok(*w), "w of {:?} (N = {})", a, n);
        }
    }

    /// Local factors match the derivation at multiplicative primes:
    /// split → −1, non-split → +1, good → +1.
    #[test]
    fn test_local_root_number_multiplicative() {
        // 15a1: non-split I4 at 3 (+1), split I4 at 5 (−1).
        let e = curve(1, 1, 1, -10, -10);
        assert_eq!(e.local_root_number(&Integer::from(3)), Ok(1));
        assert_eq!(e.local_root_number(&Integer::from(5)), Ok(-1));
        assert_eq!(e.local_root_number(&Integer::from(7)), Ok(1), "good prime");
    }
}
