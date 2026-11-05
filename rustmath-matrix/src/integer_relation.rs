//! Real integer-relation detection via LLL (an HJLS/PSLQ-style finder).
//!
//! Given a real vector `x = (x_1, …, x_n)` (arbitrary-precision `rug::Float`),
//! [`integer_relation`] searches for a nonzero integer vector `a` with
//! `Σ a_i x_i = 0` to the working precision, returning `None` when no genuine
//! small-coefficient relation exists.
//!
//! Method (classic LLL integer-relation lattice, e.g. Cohen, *A Course in
//! Computational Algebraic Number Theory*, §2.7.2): build the `n × (n+1)`
//! integer lattice whose `i`-th row is the unit vector `e_i` followed by the
//! rounded, scaled coordinate `round(W · x_i)` with `W = 2^e`. Reducing this
//! lattice pulls forward the shortest vector; a genuine relation `a` yields the
//! vector `(a, W·(Σ a_i x_i) + ρ)` whose last coordinate is tiny (the residual
//! `Σ a_i x_i ≈ 0` and the rounding term `ρ = Σ a_i·frac(W x_i)` are both
//! `O(‖a‖)`), so its norm is just `‖a‖` — dramatically shorter than any
//! non-relation, whose last coordinate is forced to `~W^{1/n}` by Minkowski's
//! bound. The `e_i` prefix records the integer combination, so the relation is
//! read straight off the reduced row.
//!
//! ## Why an exact-integer LLL (not the crate's `lll::lll_reduce`)
//! The crate's [`crate::lll::lll_reduce`] computes its Gram–Schmidt in `f64`.
//! With a weight `W = 2^{300}` the lattice entries are ~300-bit integers, and
//! the size-reduction quotients `μ_{k,j}` then need hundreds of bits to round
//! correctly — far beyond `f64`'s 53. Empirically that reduction stalls and
//! returns garbage (coefficients ~10^45, relation never surfaced). Integer
//! relation detection therefore uses the exact-arithmetic integer LLL of Cohen
//! (Algorithm 2.6.3): the Gram–Schmidt data is carried as exact integers
//! `d_k` (Gram determinants) and `λ_{k,j}`, so every rounding is exact and the
//! reduction is correct at any precision.
//!
//! ## Choice of the weight exponent `e`
//! We take `e = floor(prec_bits · 3/4)`, i.e. `W = 2^{⌊3·prec/4⌋}`. Two
//! competing error budgets fix the fraction:
//!   * **Residual guard** — for a true relation the stored reals annihilate only
//!     up to their own precision, `|Σ a_i x_i| ≈ ‖a‖·2^{-prec}`. Its scaled
//!     contribution to the last lattice coordinate is `W·|Σ a_i x_i| ≈
//!     ‖a‖·2^{e-prec} = ‖a‖·2^{-prec/4}`, which must stay `≪ 1` so the true
//!     relation's vector really is short. The unused `1/4` of precision is that
//!     guard (100 bits at 400-bit precision).
//!   * **Separation** — a *non*-relation cannot get its last coordinate below
//!     the Minkowski floor `~W^{1/n} = 2^{e/n}`, so its coefficients are
//!     `~2^{e/n}`. Using most of the precision for `W` (numerator `3`) keeps
//!     that floor astronomically far above genuine coefficient sizes.
//!
//! ## Acceptance (a candidate is returned only if BOTH hold)
//!   1. **Coefficient bound** — `‖a‖_∞ < 2^{e/2n}`, safely below the `2^{e/n}`
//!      Minkowski floor, so the large-coefficient vector LLL is forced to return
//!      for independent inputs is rejected.
//!   2. **Verified residual** — `|Σ a_i x_i|`, recomputed in high precision from
//!      the *original* reals, is below `2^{-e} = 1/W`. A true relation clears
//!      this by the full guard (`2^{-prec} ≪ 2^{-e}`); a spurious vector's
//!      residual sits at the `~2^{-e·(n-1)/n}` floor, above the bound.
//!
//! This is the key adversarial guarantee: for algebraically independent reals no
//! bounded-coefficient combination approaches `0` to `2^{-e}`, so `None` is
//! returned rather than a fabricated relation.

use num_bigint::{BigInt, Sign};
use num_traits::{Signed, Zero};
use rug::Float;
use rustmath_integers::Integer;

/// `W = 2^{⌊prec · WEIGHT_NUM / WEIGHT_DEN⌋}`. See the module docs for why 3/4.
const WEIGHT_NUM: u64 = 3;
const WEIGHT_DEN: u64 = 4;

// ---------------------------------------------------------------------------
// Type conversions
// ---------------------------------------------------------------------------

/// `num_bigint::BigInt` → `rug::Integer`, transferring magnitude bytes directly.
fn bigint_to_rug(n: &BigInt) -> rug::Integer {
    let (sign, bytes) = n.to_bytes_le();
    let mag = rug::Integer::from_digits(&bytes, rug::integer::Order::Lsf);
    if sign == Sign::Minus {
        -mag
    } else {
        mag
    }
}

/// `rug::Integer` → `num_bigint::BigInt` via an exact decimal string. (Uses the
/// non-`std`-gated `Display`, so it works with rug built without default
/// features, matching the rest of the workspace.)
fn rug_to_bigint(z: &rug::Integer) -> BigInt {
    let s = z.to_string();
    BigInt::parse_bytes(s.as_bytes(), 10).expect("rug integer decimal string parses as BigInt")
}

// ---------------------------------------------------------------------------
// Exact-integer LLL (Cohen, Algorithm 2.6.3), δ = 3/4
// ---------------------------------------------------------------------------

fn dot(a: &[Integer], b: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        s = s + x.clone() * y.clone();
    }
    s
}

/// Floor of `x / d` for `d > 0` (`BigInt` division truncates toward zero).
fn floor_div(x: &Integer, d: &Integer) -> Integer {
    let t = x.clone() / d.clone();
    let r = x.clone() - t.clone() * d.clone();
    if r.signum() < 0 {
        t - Integer::one()
    } else {
        t
    }
}

/// Round `a / b` to the nearest integer (`b > 0`; ties toward `+∞`, which still
/// leaves `|a/b − q| ≤ 1/2`).
fn round_div(a: &Integer, b: &Integer) -> Integer {
    let two = Integer::from(2);
    let num = a.clone() * two.clone() + b.clone();
    let den = b.clone() * two;
    floor_div(&num, &den)
}

/// LLL-reduce `basis` (rows, each in `ℤ^m`, linearly independent) in place using
/// exact integer Gram–Schmidt. Returns the reduced rows. `δ = 3/4`.
fn lll_reduce_exact(basis: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let n = basis.len();
    let mut b: Vec<Vec<Integer>> = basis.to_vec();
    if n <= 1 {
        return b;
    }

    // d[0..=n]: d[i] = Gram determinant of the first i rows; d[0] = 1.
    let mut d: Vec<Integer> = vec![Integer::zero(); n + 1];
    d[0] = Integer::one();
    // lam[k][j] for 1 ≤ j < k ≤ n (1-indexed rows; row k is b[k-1]).
    let mut lam: Vec<Vec<Integer>> = vec![vec![Integer::zero(); n + 1]; n + 1];

    // b1·b1
    d[1] = dot(&b[0], &b[0]);

    // Size-reduce row k against row l (1-indexed).
    fn red(
        k: usize,
        l: usize,
        b: &mut [Vec<Integer>],
        lam: &mut [Vec<Integer>],
        d: &[Integer],
    ) {
        if (lam[k][l].clone() * Integer::from(2)).abs() <= d[l] {
            return;
        }
        let q = round_div(&lam[k][l], &d[l]);
        if q.is_zero() {
            return;
        }
        // b_k -= q · b_l
        let m = b[k - 1].len();
        for c in 0..m {
            b[k - 1][c] = b[k - 1][c].clone() - q.clone() * b[l - 1][c].clone();
        }
        lam[k][l] = lam[k][l].clone() - q.clone() * d[l].clone();
        for i in 1..l {
            lam[k][i] = lam[k][i].clone() - q.clone() * lam[l][i].clone();
        }
    }

    let mut kmax = 1usize;
    let mut k = 2usize;
    // Exact LLL always terminates; the cap only guards against a coding slip.
    let cap = 1000 * n * n + 100_000;
    let mut iters = 0usize;
    while k <= n {
        iters += 1;
        if iters > cap {
            break;
        }
        // Incremental exact Gram–Schmidt for a freshly reached row.
        if k > kmax {
            kmax = k;
            for j in 1..=k {
                let mut u = dot(&b[k - 1], &b[j - 1]);
                for i in 1..j {
                    u = (d[i].clone() * u - lam[k][i].clone() * lam[j][i].clone())
                        / d[i - 1].clone();
                }
                if j < k {
                    lam[k][j] = u;
                } else {
                    d[k] = u;
                }
            }
        }

        red(k, k - 1, &mut b, &mut lam, &d);

        // Lovász test (δ = 3/4): swap iff 4·d_k·d_{k-2} < 3·d_{k-1}² − 4·λ².
        let lhs = Integer::from(4) * d[k].clone() * d[k - 2].clone();
        let rhs = Integer::from(3) * d[k - 1].clone() * d[k - 1].clone()
            - Integer::from(4) * lam[k][k - 1].clone() * lam[k][k - 1].clone();
        if lhs < rhs {
            // SWAP(k)
            b.swap(k - 1, k - 2);
            for j in 1..=(k - 2) {
                let tmp = lam[k][j].clone();
                lam[k][j] = lam[k - 1][j].clone();
                lam[k - 1][j] = tmp;
            }
            let lambda = lam[k][k - 1].clone();
            let bb = (d[k - 2].clone() * d[k].clone() + lambda.clone() * lambda.clone())
                / d[k - 1].clone();
            for i in (k + 1)..=kmax {
                let t = lam[i][k].clone();
                lam[i][k] = (d[k].clone() * lam[i][k - 1].clone() - lambda.clone() * t.clone())
                    / d[k - 1].clone();
                lam[i][k - 1] =
                    (bb.clone() * t + lambda.clone() * lam[i][k].clone()) / d[k].clone();
            }
            d[k - 1] = bb;
            k = if k > 2 { k - 1 } else { 2 };
        } else {
            let mut l = k as isize - 2;
            while l >= 1 {
                red(k, l as usize, &mut b, &mut lam, &d);
                l -= 1;
            }
            k += 1;
        }
    }
    b
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Euclidean gcd of magnitudes.
fn bigint_gcd(a: &BigInt, b: &BigInt) -> BigInt {
    let mut a = a.abs();
    let mut b = b.abs();
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

/// Divide out the content and fix the sign so the first nonzero entry is
/// positive — a canonical representative of the relation up to sign/scale.
fn normalize(a: &[BigInt]) -> Vec<BigInt> {
    let mut g = BigInt::zero();
    for c in a {
        g = bigint_gcd(&g, c);
    }
    if g.is_zero() {
        return a.to_vec();
    }
    let mut out: Vec<BigInt> = a.iter().map(|c| c / &g).collect();
    if let Some(first) = out.iter().find(|c| !c.is_zero()) {
        if first.sign() == Sign::Minus {
            out = out.iter().map(|c| -c).collect();
        }
    }
    out
}

/// Search for a nonzero integer relation `a` with `Σ a_i x_i = 0` to precision.
///
/// Returns `Some(a)` (a primitive vector with canonical sign) only when a
/// genuine small-coefficient relation is found and independently verified in
/// high precision; otherwise `None`. `prec_bits` is the working precision that
/// fixes the lattice weight `W = 2^{⌊3·prec_bits/4⌋}`; pass the precision at
/// which the `x_i` are known.
pub fn integer_relation(x: &[Float], prec_bits: u32) -> Option<Vec<BigInt>> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    if x.iter().any(|xi| !xi.is_finite()) {
        return None;
    }

    // Weight exponent e (W = 2^e). Needs to be positive to separate at all.
    let e: u32 = ((prec_bits as u64) * WEIGHT_NUM / WEIGHT_DEN).try_into().ok()?;
    if e == 0 {
        return None;
    }

    // Working precision for the exact integer round(W·x_i): the shift `<< e`
    // (exact) plus the input's own bits plus slack; rounding here loses nothing.
    let work_prec = e.saturating_add(prec_bits).saturating_add(64);

    // Build the n × (n+1) lattice: row i = e_i ++ [ round(W · x_i) ].
    let mut basis: Vec<Vec<Integer>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<Integer> = (0..n)
            .map(|j| if i == j { Integer::one() } else { Integer::zero() })
            .collect();
        let scaled = Float::with_val(work_prec, &x[i] << e).round();
        let zi = scaled.to_integer()?; // finite ⇒ Some
        row.push(Integer::from(rug_to_bigint(&zi)));
        basis.push(row);
    }

    let reduced = lll_reduce_exact(&basis);

    // Acceptance thresholds.
    // Coefficient bound: below the ~2^{e/n} Minkowski floor for non-relations.
    let max_coeff_bits: u64 = ((e as u64) / (2 * n as u64)).max(8);
    // Residual bound: 2^{-e} = 1/W.
    let res_prec = prec_bits.saturating_add(64).max(2);
    let bound = Float::with_val(res_prec, 1) >> e;

    for row in &reduced {
        let a: Vec<BigInt> = row[..n].iter().map(|z| z.as_bigint().clone()).collect();
        if a.iter().all(|c| c.is_zero()) {
            continue;
        }
        // Gate 1: coefficient size.
        if a.iter().any(|c| c.bits() > max_coeff_bits) {
            continue;
        }
        // Gate 2: high-precision residual, recomputed from the ORIGINAL reals.
        let mut s = Float::with_val(res_prec, 0);
        for (ai, xi) in a.iter().zip(x.iter()) {
            let af = Float::with_val(res_prec, bigint_to_rug(ai));
            s += Float::with_val(res_prec, &af * xi);
        }
        if s.abs() < bound {
            return Some(normalize(&a));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical(a: &[i64]) -> Vec<BigInt> {
        normalize(&a.iter().map(|&c| BigInt::from(c)).collect::<Vec<_>>())
    }

    /// Independently verify (in higher precision than the finder used) that a
    /// returned relation genuinely annihilates the reals: returns the binary
    /// exponent of `|Σ a_i x_i|` (very negative ⇒ near zero).
    fn residual_exp(a: &[BigInt], x: &[Float], prec: u32) -> i32 {
        let mut s = Float::with_val(prec + 128, 0);
        for (ai, xi) in a.iter().zip(x.iter()) {
            let af = Float::with_val(prec + 128, bigint_to_rug(ai));
            s += Float::with_val(prec + 128, &af * xi);
        }
        let abs = s.abs();
        if abs == 0 {
            i32::MIN
        } else {
            abs.get_exp().unwrap_or(0)
        }
    }

    #[test]
    fn recovers_planted_relation() {
        // Independent oracle: the relation a = [3,-5,2,7,-1,4] is PLANTED by
        // construction (r6 is chosen to make it hold exactly to 400 bits); the
        // expected answer is therefore known a priori, not read from the code.
        // Cross-checked with mpmath.pslq → [-3,5,-2,-7,1,-4] = -a.
        let prec = 400u32;
        let r1 = Float::with_val(prec, 2).sqrt();
        let r2 = Float::with_val(prec, 3).sqrt();
        let r3 = Float::with_val(prec, 5).sqrt();
        let r4 = Float::with_val(prec, 7).sqrt();
        let r5 = Float::with_val(prec, 11).sqrt();
        // combo = 3 r1 - 5 r2 + 2 r3 + 7 r4 - r5
        let combo = Float::with_val(prec, 3 * &r1)
            - Float::with_val(prec, 5 * &r2)
            + Float::with_val(prec, 2 * &r3)
            + Float::with_val(prec, 7 * &r4)
            - &r5;
        // r6 = -combo/4 (division by 4 is an exact binary shift), so
        // 3 r1 - 5 r2 + 2 r3 + 7 r4 - r5 + 4 r6 = 0 to full precision.
        let r6 = Float::with_val(prec, -&combo) / 4;
        let x = vec![r1, r2, r3, r4, r5, r6];

        let rel = integer_relation(&x, prec).expect("planted relation must be found");
        assert_eq!(rel, canonical(&[3, -5, 2, 7, -1, 4]));
        // Independent high-precision verification of the returned relation.
        assert!(
            residual_exp(&rel, &x, prec) < -(prec as i32) / 2,
            "returned relation does not annihilate the reals in HP"
        );
    }

    #[test]
    fn rejects_independent_reals() {
        // KEY ADVERSARIAL CASE. √2,√3,√5,√7,√11,√13 are Q-linearly independent
        // (classical), so NO integer relation exists. A finder that invents one
        // is broken. mpmath.pslq also returns None here.
        let prec = 400u32;
        let x: Vec<Float> = [2, 3, 5, 7, 11, 13]
            .iter()
            .map(|&p| Float::with_val(prec, p).sqrt())
            .collect();
        assert!(
            integer_relation(&x, prec).is_none(),
            "fabricated a relation among independent reals"
        );
    }

    #[test]
    fn recovers_simple_scaling_relation() {
        // x = [√2, 2√2]; 2·x0 - x1 = 0 exactly (2√2 = √2 << 1). Expect [2,-1].
        let prec = 300u32;
        let a = Float::with_val(prec, 2).sqrt();
        let b = Float::with_val(prec, 2 * &a);
        let x = vec![a, b];
        let rel = integer_relation(&x, prec).expect("scaling relation must be found");
        assert_eq!(rel, canonical(&[2, -1]));
    }

    #[test]
    fn recovers_rational_multiple_relation() {
        // A 3-term planted relation with a different signature: pick π and e as
        // independent reals, r3 = (2π - 3e)/5, so 2π - 3e - 5 r3 = 0. Expect
        // [2,-3,-5].
        let prec = 350u32;
        let pi = Float::with_val(prec, rug::float::Constant::Pi);
        let ee = Float::with_val(prec, 1).exp();
        let combo = Float::with_val(prec, 2 * &pi) - Float::with_val(prec, 3 * &ee);
        let r3 = Float::with_val(prec, &combo / 5);
        let x = vec![pi, ee, r3];
        let rel = integer_relation(&x, prec).expect("planted 3-term relation must be found");
        assert_eq!(rel, canonical(&[2, -3, -5]));
    }

    #[test]
    fn rejects_two_independent_reals() {
        // √2, √3: independent, no 2-term relation.
        let prec = 300u32;
        let x = vec![
            Float::with_val(prec, 2).sqrt(),
            Float::with_val(prec, 3).sqrt(),
        ];
        assert!(integer_relation(&x, prec).is_none());
    }

    #[test]
    fn rejects_three_independent_reals() {
        // √2, √3, √5: independent, no relation.
        let prec = 350u32;
        let x: Vec<Float> = [2, 3, 5]
            .iter()
            .map(|&p| Float::with_val(prec, p).sqrt())
            .collect();
        assert!(integer_relation(&x, prec).is_none());
    }

    #[test]
    fn empty_input_is_none() {
        assert!(integer_relation(&[], 400).is_none());
    }
}
