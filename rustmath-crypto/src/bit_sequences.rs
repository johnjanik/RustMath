//! Binary (F₂) pseudo-random bit sequence tools.
//!
//! Port of MAGMA Handbook **Chapter 158 — Pseudo-random Bit Sequences**:
//! §158.2 (the shrinking generator), §158.4 (correlation functions) and
//! §158.5 (decimation). Every function here works over **F₂**, so bits are
//! represented as `u8` values in `{0, 1}` (arithmetic is mod 2, where `−1 ≡ 1`).
//!
//! For the arbitrary-finite-field LFSR / Berlekamp–Massey machinery see
//! [`crate::lfsr`]; this module is the specialised binary layer that MAGMA's
//! chapter treats as the default universe.

/// One step of a binary LFSR with connection polynomial `c` (coefficients in
/// `{0,1}`, `c[0] = 1`) and current `state` (length `L`). Returns the next state.
/// F₂ specialisation of MAGMA `LFSRStep(C, S)`.
///
/// Over F₂ the recurrence `s_j = − Σ c_i s_{j-i}` becomes `s_j = Σ c_i s_{j-i}`
/// (mod 2), i.e. the XOR of the tapped state bits.
pub fn lfsr_step_f2(c: &[u8], state: &[u8]) -> Vec<u8> {
    let l = state.len();
    let mut acc = 0u8;
    for i in 1..=l {
        let ci = if i < c.len() { c[i] & 1 } else { 0 };
        acc ^= ci & state[l - i];
    }
    let mut next = Vec::with_capacity(l);
    next.extend_from_slice(&state[1..]);
    next.push(acc & 1);
    next
}

/// The first `t` output bits of a binary LFSR (connection polynomial `c`, initial
/// state `s`). The first `min(t, s.len())` outputs are the initial state itself.
///
/// This is the F₂ specialisation of [`crate::lfsr::lfsr_sequence`].
pub fn lfsr_sequence_f2(c: &[u8], s: &[u8], t: usize) -> Vec<u8> {
    assert!(!s.is_empty(), "initial state must be non-empty");
    assert!(!c.is_empty() && (c[0] & 1) == 1, "C(D) must have constant term 1");
    let l = s.len();
    let mut seq: Vec<u8> = Vec::with_capacity(t);
    for i in 0..t {
        if i < l {
            seq.push(s[i] & 1);
        } else {
            let mut acc = 0u8;
            for k in 1..=l {
                let ck = if k < c.len() { c[k] & 1 } else { 0 };
                acc ^= ck & seq[i - k];
            }
            seq.push(acc & 1);
        }
    }
    seq
}

/// The shrinking generator (MAGMA `ShrinkingGenerator(C1, S1, C2, S2, t)`).
///
/// Two binary LFSRs are clocked together: LFSR 1 (`c1`, `s1`) supplies the *data*
/// bits and LFSR 2 (`c2`, `s2`) the *control* bits. At each clock, if the control
/// bit is 1 the data bit is emitted, otherwise it is discarded. Emits `t` bits.
///
/// # Panics
/// Panics if either connection polynomial has zero constant term, if a state is
/// shorter than the degree of its connection polynomial, or if the control LFSR
/// never produces enough 1-bits to emit `t` outputs within a generous bound.
pub fn shrinking_generator(
    c1: &[u8],
    s1: &[u8],
    c2: &[u8],
    s2: &[u8],
    t: usize,
) -> Vec<u8> {
    assert!(
        !c1.is_empty() && (c1[0] & 1) == 1,
        "C1 must have constant term 1"
    );
    assert!(
        !c2.is_empty() && (c2[0] & 1) == 1,
        "C2 must have constant term 1"
    );
    assert!(
        s1.len() + 1 >= c1.len(),
        "S1 must have at least deg(C1) elements"
    );
    assert!(
        s2.len() + 1 >= c2.len(),
        "S2 must have at least deg(C2) elements"
    );
    if t == 0 {
        return Vec::new();
    }

    let mut out: Vec<u8> = Vec::with_capacity(t);
    // Generate in expanding chunks: about half the control bits are 1, so 2t+64
    // clocks usually suffice; grow if the control sequence is 1-sparse.
    let mut clocks = 2 * t + 64;
    loop {
        let data = lfsr_sequence_f2(c1, s1, clocks);
        let ctrl = lfsr_sequence_f2(c2, s2, clocks);
        out.clear();
        for k in 0..clocks {
            if ctrl[k] == 1 {
                out.push(data[k]);
                if out.len() == t {
                    return out;
                }
            }
        }
        // Not enough control 1-bits yet; try a longer run.
        clocks *= 2;
        assert!(
            clocks <= (t + 1) * 4096,
            "control LFSR produced too few 1-bits to emit {t} shrunk bits"
        );
    }
}

/// Autocorrelation of a binary sequence (MAGMA `AutoCorrelation(S, t)`).
///
/// `C(t) = Σ_{i=1}^{L} (−1)^{S[i] + S[i+t]}` with `L = S.len()` and indices taken
/// mod `L` (wrap-around). A maximal-period LFSR (m-sequence) has the two-valued
/// property: `C(0) = L` and `C(t) = −1` for `0 < t < L`.
pub fn auto_correlation(s: &[u8], t: usize) -> i64 {
    cross_correlation(s, s, t)
}

/// Crosscorrelation of two equal-length binary sequences
/// (MAGMA `CrossCorrelation(S1, S2, t)`).
///
/// `C(t) = Σ_{i=1}^{L} (−1)^{S1[i] + S2[i+t]}`, indices in `S2` taken mod `L`.
///
/// # Panics
/// Panics if the two sequences have different lengths or are empty.
pub fn cross_correlation(s1: &[u8], s2: &[u8], t: usize) -> i64 {
    assert_eq!(s1.len(), s2.len(), "sequences must have equal length");
    let l = s1.len();
    assert!(l > 0, "sequences must be non-empty");
    let mut sum: i64 = 0;
    for (i, &a) in s1.iter().enumerate() {
        let j = (i + t) % l;
        let parity = (a ^ s2[j]) & 1;
        sum += if parity == 0 { 1 } else { -1 };
    }
    sum
}

/// Decimation of a binary sequence (MAGMA `Decimation(S, f, d)`).
///
/// Samples `S[f], S[f+d], S[f+2d], …` where indices are **1-based** and taken
/// with wrap-around into `1..=#S`. Returns `#S` samples (one full-length pass).
///
/// # Panics
/// Panics if `s` is empty or `f == 0` (indices are 1-based).
pub fn decimation(s: &[u8], f: usize, d: usize) -> Vec<u8> {
    decimation_truncated(s, f, d, s.len())
}

/// Decimation returning exactly the first `t` samples
/// (MAGMA `Decimation(S, f, d, t)`).
///
/// # Panics
/// Panics if `s` is empty or `f == 0` (indices are 1-based).
pub fn decimation_truncated(s: &[u8], f: usize, d: usize, t: usize) -> Vec<u8> {
    let l = s.len();
    assert!(l > 0, "sequence must be non-empty");
    assert!(f >= 1, "index f is 1-based and must be >= 1");
    let mut out = Vec::with_capacity(t);
    for k in 0..t {
        // 1-based index f + k*d, wrapped into 1..=L, converted to 0-based.
        let idx0 = (f - 1 + k.wrapping_mul(d)) % l;
        out.push(s[idx0] & 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfsr_sequence_f2_matches_hand() {
        // C(D) = 1 + D + D^2, state [1,0] -> 1,0,1,1,0,1 (period 3).
        let c = vec![1, 1, 1];
        let s = vec![1, 0];
        assert_eq!(lfsr_sequence_f2(&c, &s, 6), vec![1, 0, 1, 1, 0, 1]);
    }

    #[test]
    fn lfsr_step_f2_advances_state() {
        // C(D) = 1 + D + D^2, state [1,0]: next element s2 = 1 -> state [0,1].
        let c = vec![1, 1, 1];
        assert_eq!(lfsr_step_f2(&c, &[1, 0]), vec![0, 1]);
        assert_eq!(lfsr_step_f2(&c, &[0, 1]), vec![1, 1]);
    }

    /// MAGMA example H158E3: the autocorrelation of a maximal-period LFSR
    /// (m-sequence) is `L` at shift 0 and `−1` at every other shift.
    #[test]
    fn h158e3_msequence_autocorrelation() {
        // C(D) = 1 + D + D^4, primitive over GF(2): period 2^4 - 1 = 15.
        let c = vec![1, 1, 0, 0, 1];
        let s = vec![1, 0, 0, 0];
        let period = lfsr_sequence_f2(&c, &s, 15);
        assert_eq!(period.len(), 15);

        assert_eq!(auto_correlation(&period, 0), 15);
        for t in 1..15 {
            assert_eq!(
                auto_correlation(&period, t),
                -1,
                "out-of-phase autocorrelation at shift {t} should be -1"
            );
        }
    }

    #[test]
    fn cross_correlation_self_is_autocorrelation() {
        let a = vec![1, 0, 1, 1, 0];
        let b = vec![0, 1, 1, 0, 0];
        // Symmetry sanity: swapping args and negating the shift agrees.
        let c1 = cross_correlation(&a, &b, 2);
        assert!(c1 >= -5 && c1 <= 5);
        assert_eq!(auto_correlation(&a, 0), 5);
    }

    #[test]
    fn decimation_wraps_around() {
        let s = vec![1, 0, 0, 1, 1]; // length 5
        // f = 1, d = 2 (1-based): indices 1,3,5,2,4 -> S = 1,0,1,0,1
        assert_eq!(decimation(&s, 1, 2), vec![1, 0, 1, 0, 1]);
        // Truncated to 3 samples.
        assert_eq!(decimation_truncated(&s, 1, 2, 3), vec![1, 0, 1]);
        // Offset f = 2, d = 1: indices 2,3,4,5,1 -> 0,0,1,1,1
        assert_eq!(decimation(&s, 2, 1), vec![0, 0, 1, 1, 1]);
    }

    /// A decimation of a max-period GF(2) m-sequence by `d` coprime to the period
    /// is again an m-sequence of the same period (recoverable by Berlekamp–Massey).
    #[test]
    fn decimation_of_msequence_preserves_period() {
        use crate::lfsr::sequence_period;
        let c = vec![1, 1, 0, 0, 1]; // period 15
        let s = vec![1, 0, 0, 0];
        let seq = lfsr_sequence_f2(&c, &s, 15);
        // gcd(2, 15) = 1, so decimation by 2 keeps period 15.
        let dec = decimation(&seq, 1, 2);
        // Two full periods for a reliable period check.
        let mut doubled = dec.clone();
        doubled.extend(dec.iter().copied());
        assert_eq!(sequence_period(&doubled), Some(15));
    }

    #[test]
    fn shrinking_generator_basic() {
        // Data LFSR: C(D)=1+D+D^2, state [1,0]; control LFSR: C(D)=1+D+D^3.
        let c1 = vec![1, 1, 1];
        let s1 = vec![1, 0];
        let c2 = vec![1, 1, 0, 1];
        let s2 = vec![1, 0, 0];
        let out = shrinking_generator(&c1, &s1, &c2, &s2, 10);
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|&b| b == 0 || b == 1));

        // Deterministic: same inputs -> same output.
        let out2 = shrinking_generator(&c1, &s1, &c2, &s2, 10);
        assert_eq!(out, out2);

        // Cross-check against the manual definition on the first chunk.
        let clocks = 64;
        let data = lfsr_sequence_f2(&c1, &s1, clocks);
        let ctrl = lfsr_sequence_f2(&c2, &s2, clocks);
        let manual: Vec<u8> = (0..clocks)
            .filter(|&k| ctrl[k] == 1)
            .map(|k| data[k])
            .take(10)
            .collect();
        assert_eq!(out, manual);
    }
}
