//! Field-generic Reed-Solomon codes with a real algebraic decoder
//!
//! [`GenericReedSolomonCode<F>`] is an `[n, k, n-k+1]` Reed-Solomon code over
//! any field `F` containing an element `alpha` of multiplicative order
//! exactly `n` (for `F = GF(q)` this requires `n | q - 1`; `n = q - 1` uses a
//! primitive element). Two encodings of the *same* code are provided:
//!
//! * **evaluation**: `c = (f(alpha^0), f(alpha^1), ..., f(alpha^{n-1}))` for
//!   the message polynomial `f` of degree `< k`;
//! * **systematic** (cyclic-code view): `c(x) = x^{n-k} m(x) - (x^{n-k} m(x)
//!   mod g(x))` with `g(x) = prod_{j=1}^{n-k} (x - alpha^j)`, laid out as
//!   `[parity | message]`.
//!
//! Both encodings produce words of the cyclic MDS code
//! `{c : c(alpha^j) = 0 for j = 1..n-k}` (the two views coincide because
//! `alpha` has order exactly `n`), so decoding is shared:
//!
//! 1. syndromes `S_j = r(alpha^j)`, `j = 1..2t`;
//! 2. **Berlekamp-Massey** (from `rustmath-matrix`) for the error locator
//!    `Lambda(x)` with roots at the inverse error locations;
//! 3. **Chien search** over `alpha^{-i}` for the error positions;
//! 4. **Forney's formula** `e_i = -Omega(X_i^{-1}) / Lambda'(X_i^{-1})` with
//!    `Omega(x) = S(x) Lambda(x) mod x^{2t}` for the error values;
//! 5. a **re-encode self-check**: all `n - k` syndromes of the corrected word
//!    are recomputed and must vanish, so a beyond-capacity pattern that
//!    slips through steps 2-4 is still reported as an error whenever the
//!    syndromes expose it. (Bounded-distance decoding caveat: if more than
//!    `t` errors move the received word to within distance `t` of a
//!    *different* codeword, that codeword is returned — no decoder can
//!    distinguish this case.)
//!
//! The module also provides honest primitive-element searches (factor the
//! group order, check `g^{(q-1)/r} != 1` for every prime `r | q - 1` — no
//! hardcoded tables).

use rustmath_core::{Field, NumericConversion, Ring};
use rustmath_finitefields::{FiniteField, FiniteFieldElement};
use rustmath_integers::prime::factor;
use rustmath_integers::Integer;
use rustmath_matrix::berlekamp_massey::berlekamp_massey;

/// A Reed-Solomon `[n, k, n-k+1]` code over a field `F`, defined by an
/// element `alpha` of multiplicative order exactly `n`.
#[derive(Clone, Debug)]
pub struct GenericReedSolomonCode<F: Field> {
    n: usize,
    k: usize,
    /// `alpha^0 .. alpha^{n-1}` (all distinct since `alpha` has order `n`).
    alpha_pows: Vec<F>,
    /// Bound multiplicative identity (`alpha * alpha^{-1}`), usable where the
    /// unbound `F::one()` sentinel of the finite-field types would not be.
    one: F,
    /// Bound additive identity (`alpha - alpha`).
    zero: F,
    /// `n^{-1}` in `F`, for message recovery by inverse DFT.
    n_inv: F,
    /// `g(x) = prod_{j=1}^{n-k} (x - alpha^j)`, little-endian, monic.
    generator_poly: Vec<F>,
}

/// `base^e` by square-and-multiply, with a caller-supplied bound `one`.
fn fpow<F: Field>(base: &F, mut e: usize, one: &F) -> F {
    let mut result = one.clone();
    let mut b = base.clone();
    while e > 0 {
        if e & 1 == 1 {
            result = result * b.clone();
        }
        b = b.clone() * b.clone();
        e >>= 1;
    }
    result
}

/// Evaluate a little-endian polynomial at `x` (Horner), seeded with `zero`.
fn poly_eval<F: Field>(coeffs: &[F], x: &F, zero: &F) -> F {
    let mut acc = zero.clone();
    for c in coeffs.iter().rev() {
        acc = acc * x.clone() + c.clone();
    }
    acc
}

impl<F: Field> GenericReedSolomonCode<F> {
    /// Build the code from `alpha`, verifying that `alpha` has
    /// multiplicative order *exactly* `n` (via the prime factorization of
    /// `n` from `rustmath-integers` — `alpha^n = 1` and `alpha^{n/r} != 1`
    /// for every prime `r | n`).
    pub fn new(n: usize, k: usize, alpha: F) -> Result<Self, String> {
        if k == 0 {
            return Err("dimension k must be >= 1".to_string());
        }
        if k >= n {
            return Err(format!("need k < n, got k = {k}, n = {n}"));
        }
        if alpha.is_zero() {
            return Err("alpha must be nonzero".to_string());
        }

        // Bound identities derived from alpha (see field docs).
        let one = alpha.clone()
            * alpha
                .inverse()
                .map_err(|e| format!("alpha has no inverse: {e:?}"))?;
        let zero = alpha.clone() - alpha.clone();

        // Order check: alpha^n == 1 ...
        if fpow(&alpha, n, &one) != one {
            return Err(format!("alpha^{n} != 1: alpha does not have order {n}"));
        }
        // ... and alpha^(n/r) != 1 for every prime r | n.
        for (prime, _) in factor(&Integer::from(n as u64)) {
            let r = prime
                .to_u64()
                .ok_or_else(|| "prime factor of n too large".to_string())?
                as usize;
            if fpow(&alpha, n / r, &one) == one {
                return Err(format!(
                    "alpha^({n}/{r}) = 1: alpha has order dividing {}, not exactly {n}",
                    n / r
                ));
            }
        }

        // alpha^0 .. alpha^{n-1}; distinct because ord(alpha) = n exactly.
        let mut alpha_pows = Vec::with_capacity(n);
        alpha_pows.push(one.clone());
        for i in 1..n {
            let next = alpha_pows[i - 1].clone() * alpha.clone();
            alpha_pows.push(next);
        }

        // n as a field element, for the inverse DFT. n | q-1 forces
        // char(F) to not divide n, but check honestly for arbitrary alpha.
        let mut n_in_field = zero.clone();
        for _ in 0..n {
            n_in_field = n_in_field + one.clone();
        }
        if n_in_field.is_zero() {
            return Err(format!(
                "the field characteristic divides n = {n}; inverse DFT is impossible"
            ));
        }
        let n_inv = n_in_field
            .inverse()
            .map_err(|e| format!("cannot invert n in F: {e:?}"))?;

        // g(x) = prod_{j=1}^{n-k} (x - alpha^j).
        let mut generator_poly = vec![one.clone()];
        for alpha_j in alpha_pows.iter().take(n - k + 1).skip(1) {
            let mut next = vec![zero.clone(); generator_poly.len() + 1];
            for (i, c) in generator_poly.iter().enumerate() {
                next[i + 1] = next[i + 1].clone() + c.clone(); // * x
                next[i] = next[i].clone() - c.clone() * alpha_j.clone(); // * (-alpha^j)
            }
            generator_poly = next;
        }

        Ok(GenericReedSolomonCode {
            n,
            k,
            alpha_pows,
            one,
            zero,
            n_inv,
            generator_poly,
        })
    }

    /// Code length `n`.
    pub fn length(&self) -> usize {
        self.n
    }

    /// Code dimension `k`.
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// Minimum distance `d = n - k + 1` (MDS: the BCH bound on the `n - k`
    /// consecutive roots `alpha^1..alpha^{n-k}` gives `d >= n - k + 1`, and
    /// the Singleton bound gives `d <= n - k + 1`).
    pub fn minimum_distance(&self) -> usize {
        self.n - self.k + 1
    }

    /// Error correction capability `t = floor((n-k)/2)`.
    pub fn error_correction_capability(&self) -> usize {
        (self.n - self.k) / 2
    }

    /// The defining element `alpha` (of multiplicative order exactly `n`).
    pub fn alpha(&self) -> &F {
        &self.alpha_pows[1]
    }

    /// `g(x) = prod_{j=1}^{n-k} (x - alpha^j)`, little-endian, monic.
    pub fn generator_polynomial(&self) -> &[F] {
        &self.generator_poly
    }

    /// Evaluation encoding: `c_i = f(alpha^i)` where `f` is the message
    /// polynomial `message[0] + message[1] x + ... + message[k-1] x^{k-1}`.
    pub fn encode_evaluation(&self, message: &[F]) -> Result<Vec<F>, String> {
        if message.len() != self.k {
            return Err(format!(
                "message length {} does not match dimension {}",
                message.len(),
                self.k
            ));
        }
        Ok(self
            .alpha_pows
            .iter()
            .map(|x| poly_eval(message, x, &self.zero))
            .collect())
    }

    /// Systematic encoding (cyclic view): codeword coefficients
    /// `[-(x^{n-k} m(x) mod g) | m]`, so the message occupies the last `k`
    /// coordinates and `g(x) | c(x)`.
    pub fn encode_systematic(&self, message: &[F]) -> Result<Vec<F>, String> {
        if message.len() != self.k {
            return Err(format!(
                "message length {} does not match dimension {}",
                message.len(),
                self.k
            ));
        }
        let r = self.n - self.k;
        // shifted = x^r * m(x)
        let mut rem: Vec<F> = vec![self.zero.clone(); r];
        rem.extend_from_slice(message);
        // reduce mod g (monic, degree r): eliminate from the top down
        for i in (r..self.n).rev() {
            let c = rem[i].clone();
            if c.is_zero() {
                continue;
            }
            let base = i - r;
            for (j, gj) in self.generator_poly.iter().enumerate() {
                rem[base + j] = rem[base + j].clone() - c.clone() * gj.clone();
            }
        }
        // codeword = x^r m(x) - rem: parity = -rem in the low r coords.
        let mut codeword: Vec<F> = rem[..r].iter().map(|c| -c.clone()).collect();
        codeword.extend_from_slice(message);
        Ok(codeword)
    }

    /// Syndromes `S_j = r(alpha^j)` for `j = 1..=count`.
    fn syndromes(&self, word: &[F], count: usize) -> Vec<F> {
        (1..=count)
            .map(|j| {
                let mut s = self.zero.clone();
                for (i, w) in word.iter().enumerate() {
                    if !w.is_zero() {
                        s = s + w.clone() * self.alpha_pows[(i * j) % self.n].clone();
                    }
                }
                s
            })
            .collect()
    }

    /// Correct up to `t` errors in `received`, returning the corrected
    /// codeword and the number of symbol errors fixed.
    ///
    /// Berlekamp-Massey + Chien search + Forney (see the module docs).
    /// Every detectable failure — locator degree out of range, root count
    /// not matching the locator degree, a zero Forney value, or nonzero
    /// post-correction syndromes — returns `Err` rather than a guess.
    pub fn correct(&self, received: &[F]) -> Result<(Vec<F>, usize), String> {
        if received.len() != self.n {
            return Err(format!(
                "received word length {} does not match code length {}",
                received.len(),
                self.n
            ));
        }
        let full = self.n - self.k;
        let t = full / 2;
        let two_t = 2 * t; // even prefix fed to Berlekamp-Massey

        let all_syndromes = self.syndromes(received, full);
        if all_syndromes.iter().all(|s| s.is_zero()) {
            return Ok((received.to_vec(), 0));
        }
        if t == 0 {
            return Err(
                "nonzero syndrome with t = 0: errors detected but this code cannot correct any"
                    .to_string(),
            );
        }

        // 1. Berlekamp-Massey: minimal connection polynomial Lambda with
        //    Lambda(0) = 1 and S_i + sum_j Lambda_j S_{i-j} = 0.
        let lambda_poly = berlekamp_massey(all_syndromes[..two_t].to_vec())
            .map_err(|e| format!("Berlekamp-Massey failed: {e:?}"))?;
        let lambda: Vec<F> = lambda_poly.coefficients().to_vec();
        let v = lambda.len() - 1;
        if v == 0 {
            return Err("nonzero syndromes but degree-0 error locator: uncorrectable".to_string());
        }
        if v > t {
            return Err(format!(
                "error locator degree {v} exceeds capability t = {t}: uncorrectable"
            ));
        }

        // 2. Chien search: position i is in error iff Lambda(alpha^{-i}) = 0.
        let mut positions = Vec::new();
        for i in 0..self.n {
            let x_inv = &self.alpha_pows[(self.n - i) % self.n];
            if poly_eval(&lambda, x_inv, &self.zero).is_zero() {
                positions.push(i);
            }
        }
        if positions.len() != v {
            return Err(format!(
                "error locator of degree {v} has {} roots among the code locators: \
                 more than t = {t} errors detected",
                positions.len()
            ));
        }

        // 3. Forney: Omega(x) = S(x) Lambda(x) mod x^{2t},
        //    e_i = -Omega(X_i^{-1}) / Lambda'(X_i^{-1}).
        let mut omega = vec![self.zero.clone(); two_t];
        for (i, s) in all_syndromes[..two_t].iter().enumerate() {
            for (j, l) in lambda.iter().enumerate() {
                if i + j < two_t {
                    omega[i + j] = omega[i + j].clone() + s.clone() * l.clone();
                }
            }
        }
        // Lambda'(x): coefficient of x^{i-1} is i * lambda_i (i as a field
        // element, built by repeated addition so it reduces mod char).
        let mut lambda_deriv = Vec::with_capacity(lambda.len() - 1);
        let mut i_in_field = self.zero.clone();
        for l in lambda.iter().skip(1) {
            i_in_field = i_in_field.clone() + self.one.clone();
            lambda_deriv.push(i_in_field.clone() * l.clone());
        }

        let mut corrected = received.to_vec();
        for &i in &positions {
            let x_inv = &self.alpha_pows[(self.n - i) % self.n];
            let num = poly_eval(&omega, x_inv, &self.zero);
            let den = poly_eval(&lambda_deriv, x_inv, &self.zero);
            if den.is_zero() {
                return Err("Forney denominator Lambda'(X^{-1}) = 0: uncorrectable".to_string());
            }
            let e = -(num
                * den
                    .inverse()
                    .map_err(|err| format!("cannot invert Forney denominator: {err:?}"))?);
            if e.is_zero() {
                return Err(
                    "Forney produced a zero error value at a located position: uncorrectable"
                        .to_string(),
                );
            }
            corrected[i] = corrected[i].clone() - e;
        }

        // 4. Re-encode self-check: all n - k syndromes of the corrected word
        //    must vanish, otherwise the error pattern exceeded capacity in a
        //    detectable way.
        let check = self.syndromes(&corrected, full);
        if !check.iter().all(|s| s.is_zero()) {
            return Err(
                "post-correction syndromes are nonzero: beyond-capacity error pattern detected"
                    .to_string(),
            );
        }

        Ok((corrected, v))
    }

    /// Decode an evaluation-encoded received word: correct errors, then
    /// recover the message by inverse DFT, `f_l = n^{-1} c(alpha^{-l})`.
    /// Returns the message and the number of errors corrected.
    pub fn decode_evaluation(&self, received: &[F]) -> Result<(Vec<F>, usize), String> {
        let (corrected, nerr) = self.correct(received)?;
        let mut coeffs = Vec::with_capacity(self.n);
        for l in 0..self.n {
            let x = &self.alpha_pows[(self.n - l) % self.n];
            coeffs.push(self.n_inv.clone() * poly_eval(&corrected, x, &self.zero));
        }
        // The corrected word is a codeword, so f_l = 0 for l >= k; verify
        // rather than assume (guards the order-of-alpha invariants).
        if coeffs[self.k..].iter().any(|c| !c.is_zero()) {
            return Err(
                "corrected word is not the evaluation of a degree < k polynomial: \
                 internal inconsistency"
                    .to_string(),
            );
        }
        coeffs.truncate(self.k);
        Ok((coeffs, nerr))
    }

    /// Decode a systematically encoded received word: correct errors, take
    /// the message from the last `k` coordinates, and re-encode as a final
    /// consistency check. Returns the message and the number of errors
    /// corrected.
    pub fn decode_systematic(&self, received: &[F]) -> Result<(Vec<F>, usize), String> {
        let (corrected, nerr) = self.correct(received)?;
        let message = corrected[self.n - self.k..].to_vec();
        // Re-encode check: systematic encoding is the unique codeword with
        // these top-k coefficients, so this must reproduce `corrected`.
        let reencoded = self.encode_systematic(&message)?;
        if reencoded != corrected {
            return Err(
                "re-encoded message does not reproduce the corrected word: mis-decode detected"
                    .to_string(),
            );
        }
        Ok((message, nerr))
    }
}

// ---------------------------------------------------------------------------
// Honest primitive-element searches (no hardcoded tables)
// ---------------------------------------------------------------------------

/// `base^e mod m` with `u128` intermediates.
fn pow_mod(base: u64, mut e: u64, m: u64) -> u64 {
    let mut result: u128 = 1;
    let mut b: u128 = (base % m) as u128;
    let m = m as u128;
    while e > 0 {
        if e & 1 == 1 {
            result = result * b % m;
        }
        b = b * b % m;
        e >>= 1;
    }
    result as u64
}

/// Find the smallest primitive root of `GF(p)` (`p` prime): factor `p - 1`
/// with `rustmath-integers` and return the first `g` with
/// `g^{(p-1)/r} != 1 (mod p)` for every prime `r | p - 1`.
pub fn find_primitive_root_gfp(p: u64) -> Result<u64, String> {
    if p == 2 {
        // GF(2)^* is trivial; 1 generates it.
        return Ok(1);
    }
    if !Integer::from(p).is_prime() {
        return Err(format!("{p} is not prime"));
    }
    let group = p - 1;
    let mut prime_divisors = Vec::new();
    for (q, _) in factor(&Integer::from(group)) {
        prime_divisors.push(
            q.to_u64()
                .ok_or_else(|| "prime factor does not fit in u64".to_string())?,
        );
    }
    for g in 2..p {
        if prime_divisors.iter().all(|&r| pow_mod(g, group / r, p) != 1) {
            return Ok(g);
        }
    }
    Err(format!("no primitive root found modulo {p}"))
}

/// Find a primitive element of `GF(p^n)` (a generator of the multiplicative
/// group): factor `p^n - 1` with `rustmath-integers`, then return the first
/// element (in base-`p` index order) whose order-check
/// `x^{(q-1)/r} != 1` passes for every prime `r | q - 1`.
pub fn find_primitive_element(field: &FiniteField) -> Result<FiniteFieldElement, String> {
    let group = field.order() - Integer::one();
    if group.is_zero() {
        return Err("multiplicative group is empty".to_string());
    }
    if group.is_one() {
        return Ok(field.one()); // GF(2)
    }
    let factors = factor(&group);
    let elements = crate::generic_code::finite_field_elements(field)?;
    for elt in elements.into_iter().skip(1) {
        // skip the zero element at index 0
        if elt.is_zero() {
            continue;
        }
        let primitive = factors
            .iter()
            .all(|(r, _)| !elt.pow_int(&(group.clone() / r.clone())).is_one());
        if primitive {
            // Lagrange sanity check (cheap, and guards the factorization).
            if !elt.pow_int(&group).is_one() {
                return Err("element passed order checks but x^(q-1) != 1: \
                            factorization of q - 1 is inconsistent"
                    .to_string());
            }
            return Ok(elt);
        }
    }
    Err(format!("no primitive element found in {field}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_finitefields::PrimeField;

    fn gf(v: u64, p: u64) -> PrimeField {
        PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
    }

    fn to_u64s(word: &[PrimeField]) -> Vec<u64> {
        word.iter().map(|x| x.value().to_u64().unwrap()).collect()
    }

    /// Pinned in Python: the smallest primitive roots of 17 and 7 are both 3
    /// (order checks against the factorizations 16 = 2^4 and 6 = 2 * 3).
    #[test]
    fn test_primitive_root_search() {
        assert_eq!(find_primitive_root_gfp(17).unwrap(), 3);
        assert_eq!(find_primitive_root_gfp(7).unwrap(), 3);
        assert!(find_primitive_root_gfp(15).is_err()); // not prime
    }

    /// The main gate: RS(16, 10) over GF(17), t = 3. Every value pinned by
    /// an independent Python implementation of GF(17), BM, Chien and Forney.
    #[test]
    fn test_rs_16_10_gf17_three_errors_exact_recovery() {
        let alpha = gf(3, 17);
        let rs = GenericReedSolomonCode::new(16, 10, alpha).unwrap();
        assert_eq!(rs.minimum_distance(), 7);
        assert_eq!(rs.error_correction_capability(), 3);

        // Python-pinned generator polynomial prod_{j=1}^{6} (x - 3^j) mod 17.
        assert_eq!(
            to_u64s(rs.generator_polynomial()),
            vec![5, 9, 11, 15, 1, 13, 1]
        );

        let message: Vec<PrimeField> = (1..=10).map(|v| gf(v, 17)).collect();
        let codeword = rs.encode_evaluation(&message).unwrap();
        // Python-pinned: (f(3^0), ..., f(3^15)) for f = 1 + 2x + ... + 10x^9.
        assert_eq!(
            to_u64s(&codeword),
            vec![4, 0, 15, 2, 15, 5, 9, 5, 12, 4, 10, 0, 12, 15, 3, 7]
        );

        // Inject exactly t = 3 errors at chosen positions.
        let mut received = codeword.clone();
        for (pos, val) in [(2usize, 5u64), (7, 9), (11, 1)] {
            received[pos] = received[pos].clone() + gf(val, 17);
        }
        let (corrected, nerr) = rs.correct(&received).unwrap();
        assert_eq!(nerr, 3);
        assert_eq!(corrected, codeword, "corrected word differs from codeword");

        let (decoded, nerr2) = rs.decode_evaluation(&received).unwrap();
        assert_eq!(nerr2, 3);
        assert_eq!(decoded, message, "message not recovered exactly");
    }

    /// t + 1 = 4 errors must fail honestly. For this pinned pattern the
    /// Python reference decoder fails in the Chien search (the degree-3
    /// locator has only 1 root among the code locators), i.e. it is
    /// detected-uncorrectable, and the Rust decoder must report Err.
    #[test]
    fn test_rs_gf17_beyond_capacity_detected() {
        let alpha = gf(3, 17);
        let rs = GenericReedSolomonCode::new(16, 10, alpha).unwrap();
        let message: Vec<PrimeField> = (1..=10).map(|v| gf(v, 17)).collect();
        let codeword = rs.encode_evaluation(&message).unwrap();
        let mut received = codeword.clone();
        for (pos, val) in [(2usize, 5u64), (7, 9), (11, 1), (14, 3)] {
            received[pos] = received[pos].clone() + gf(val, 17);
        }
        let err = rs.correct(&received).unwrap_err();
        assert!(
            err.contains("detected") || err.contains("uncorrectable"),
            "expected an honest detection failure, got: {err}"
        );
    }

    /// Battery: 10 deterministic pseudorandom error patterns of weight 1..3,
    /// all corrected exactly. The identical LCG battery was run through the
    /// independent Python pipeline first (10/10 exact recoveries).
    #[test]
    fn test_rs_gf17_error_battery() {
        let alpha = gf(3, 17);
        let rs = GenericReedSolomonCode::new(16, 10, alpha).unwrap();
        let (n, k) = (16usize, 10usize);

        let mut state: u64 = 42;
        let mut lcg = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state >> 33
        };

        for trial in 0..10 {
            let message: Vec<PrimeField> = (0..k).map(|_| gf(lcg() % 17, 17)).collect();
            let codeword = rs.encode_evaluation(&message).unwrap();
            let nerrors = 1 + (lcg() as usize) % 3;
            let mut positions: Vec<usize> = Vec::new();
            while positions.len() < nerrors {
                let p = (lcg() as usize) % n;
                if !positions.contains(&p) {
                    positions.push(p);
                }
            }
            let mut received = codeword.clone();
            for &p in &positions {
                received[p] = received[p].clone() + gf(1 + lcg() % 16, 17);
            }
            let (corrected, nerr) = rs
                .correct(&received)
                .unwrap_or_else(|e| panic!("trial {trial} failed: {e}"));
            assert_eq!(corrected, codeword, "trial {trial}: wrong correction");
            assert_eq!(nerr, nerrors, "trial {trial}: wrong error count");
        }
    }

    /// Systematic encoding over GF(7) — the old-API configuration.
    /// Python-pinned: alpha = 3, g = (x-3)(x-2) = 6 + 2x + x^2 mod 7, and
    /// message [1,2,3,4] encodes to [1, 3, 1, 2, 3, 4] (parity | message).
    #[test]
    fn test_rs_systematic_gf7() {
        let alpha = gf(3, 7);
        let rs = GenericReedSolomonCode::new(6, 4, alpha).unwrap();
        assert_eq!(to_u64s(rs.generator_polynomial()), vec![6, 2, 1]);

        let message: Vec<PrimeField> = [1u64, 2, 3, 4].iter().map(|&v| gf(v, 7)).collect();
        let codeword = rs.encode_systematic(&message).unwrap();
        assert_eq!(to_u64s(&codeword), vec![1, 3, 1, 2, 3, 4]);

        // single error (t = 1)
        let mut received = codeword.clone();
        received[0] = received[0].clone() + gf(3, 7);
        let (decoded, nerr) = rs.decode_systematic(&received).unwrap();
        assert_eq!(nerr, 1);
        assert_eq!(decoded, message);
    }

    /// alpha of the wrong order is rejected by the honest order check:
    /// 2 has order 3 in GF(7)^*, not 6.
    #[test]
    fn test_wrong_order_alpha_rejected() {
        let err = GenericReedSolomonCode::new(6, 4, gf(2, 7)).unwrap_err();
        assert!(err.contains("order"), "unexpected error: {err}");
        // and order-3 alpha is fine for n = 3
        assert!(GenericReedSolomonCode::new(3, 1, gf(2, 7)).is_ok());
    }

    // ---- extension fields ---------------------------------------------------

    fn gf9_elt(field: &FiniteField, c0: i64, c1: i64) -> FiniteFieldElement {
        field.element(vec![Integer::from(c0), Integer::from(c1)])
    }

    /// RS(8, 4) over GF(9) = GF(3)[w]/(w^2 + 2w + 2), alpha = w (the first
    /// primitive element in index order, pinned in Python along with the
    /// codeword, the 2-error recovery, and the 3-error detection).
    #[test]
    fn test_rs_gf9_extension_field_battery() {
        let field = FiniteField::new(Integer::from(3), 2).unwrap();
        let alpha = find_primitive_element(&field).unwrap();
        // Python-pinned: first primitive element is w itself (index 3).
        assert_eq!(alpha, field.generator());
        // Self-certifying order check: w^8 = 1, w^4 != 1.
        assert!(alpha.pow_int(&Integer::from(8)).is_one());
        assert!(!alpha.pow_int(&Integer::from(4)).is_one());

        let rs = GenericReedSolomonCode::new(8, 4, alpha).unwrap();
        assert_eq!(rs.error_correction_capability(), 2);

        // message [1, w, w+1, 2]
        let message = vec![
            gf9_elt(&field, 1, 0),
            gf9_elt(&field, 0, 1),
            gf9_elt(&field, 1, 1),
            gf9_elt(&field, 2, 0),
        ];
        let codeword = rs.encode_evaluation(&message).unwrap();
        // Python-pinned evaluations (c0, c1) with c = c0 + c1*w:
        let expected: Vec<FiniteFieldElement> =
            [(1, 2), (0, 2), (2, 2), (1, 2), (0, 0), (0, 1), (1, 2), (0, 1)]
                .iter()
                .map(|&(c0, c1)| gf9_elt(&field, c0, c1))
                .collect();
        assert_eq!(codeword, expected);

        // 2 errors: +w at position 1, +1 at position 6 — exact recovery.
        let mut received = codeword.clone();
        received[1] = received[1].clone() + gf9_elt(&field, 0, 1);
        received[6] = received[6].clone() + gf9_elt(&field, 1, 0);
        let (decoded, nerr) = rs.decode_evaluation(&received).unwrap();
        assert_eq!(nerr, 2);
        assert_eq!(decoded, message);

        // 3 errors (t + 1): Python reference detects (locator degree 2,
        // zero roots) — must be an honest Err here too.
        let mut received3 = codeword.clone();
        received3[1] = received3[1].clone() + gf9_elt(&field, 0, 1);
        received3[4] = received3[4].clone() + gf9_elt(&field, 2, 0);
        received3[6] = received3[6].clone() + gf9_elt(&field, 1, 0);
        assert!(rs.correct(&received3).is_err());
    }

    /// RS(7, 3) over GF(8) = GF(2)[w]/(w^3 + w + 1): 2-error recovery with
    /// the Python-pinned codeword.
    #[test]
    fn test_rs_gf8_extension_field() {
        let field = FiniteField::new(Integer::from(2), 3).unwrap();
        let alpha = find_primitive_element(&field).unwrap();
        assert!(alpha.pow_int(&Integer::from(7)).is_one());

        let elt = |c0: i64, c1: i64, c2: i64| {
            field.element(vec![Integer::from(c0), Integer::from(c1), Integer::from(c2)])
        };
        let rs = GenericReedSolomonCode::new(7, 3, alpha).unwrap();

        // message [1, w, 1 + w^2]
        let message = vec![elt(1, 0, 0), elt(0, 1, 0), elt(1, 0, 1)];
        let codeword = rs.encode_evaluation(&message).unwrap();
        // Python-pinned (c0, c1, c2) triples:
        let expected: Vec<FiniteFieldElement> = [
            (0, 1, 1),
            (1, 1, 1),
            (1, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            (0, 0, 0),
            (0, 1, 1),
        ]
        .iter()
        .map(|&(c0, c1, c2)| elt(c0, c1, c2))
        .collect();
        assert_eq!(codeword, expected);

        let mut received = codeword.clone();
        received[0] = received[0].clone() + elt(1, 1, 0);
        received[5] = received[5].clone() + elt(0, 0, 1);
        let (decoded, nerr) = rs.decode_evaluation(&received).unwrap();
        assert_eq!(nerr, 2);
        assert_eq!(decoded, message);
    }
}
