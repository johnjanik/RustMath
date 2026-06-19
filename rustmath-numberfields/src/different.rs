//! Trace form, different, and codifferent of a number field `K = ℚ[x]/(f)`.
//!
//! The trace form `T[i][j] = Tr_{K/ℚ}(ωᵢ·ωⱼ)` on the integral basis has
//! `det(T) = disc_K`. The codifferent `𝔡⁻¹ = {x : Tr(x·O_K) ⊆ ℤ}` is the trace-dual
//! lattice (columns of `T⁻¹`); the **different** `𝔡 = (𝔡⁻¹)⁻¹` is an integral ideal
//! with `N(𝔡) = |disc_K|`. Built on the ideal inverse ([`crate::ideals`]).

use crate::ideals::{ideal_from_generators, ideal_mul, make_ideal, Ideal};
use crate::round2::{bareiss_det, OrderData};
use rustmath_integers::Integer;

/// The trace-form Gram matrix `T[i][j] = Tr_{K/ℚ}(ωᵢ·ωⱼ)`. `det(T) = disc_K`.
pub fn trace_form(ord: &OrderData) -> Vec<Vec<Integer>> {
    let n = ord.n;
    // Tr(ω_k) = trace of multiply-by-ω_k = Σ_b sc[k][b][b]
    let tr_omega: Vec<Integer> = (0..n)
        .map(|k| {
            let mut s = Integer::zero();
            for b in 0..n {
                s = s + ord.sc[k][b][b].clone();
            }
            s
        })
        .collect();
    // T[i][j] = Tr(ω_i ω_j) = Σ_m sc[i][j][m]·Tr(ω_m)
    let mut t = vec![vec![Integer::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = Integer::zero();
            for m in 0..n {
                if !ord.sc[i][j][m].is_zero() {
                    s = s + ord.sc[i][j][m].clone() * tr_omega[m].clone();
                }
            }
            t[i][j] = s;
        }
    }
    t
}

fn minor(m: &[Vec<Integer>], ri: usize, ci: usize) -> Integer {
    let n = m.len();
    let sub: Vec<Vec<Integer>> = (0..n)
        .filter(|&r| r != ri)
        .map(|r| (0..n).filter(|&c| c != ci).map(|c| m[r][c].clone()).collect())
        .collect();
    bareiss_det(&sub)
}

/// Adjugate `adj(T) = det(T)·T⁻¹` (integer), `adj[k][i] = (−1)^{i+k}·minor(i,k)`.
fn adjugate(t: &[Vec<Integer>]) -> Vec<Vec<Integer>> {
    let n = t.len();
    let mut adj = vec![vec![Integer::zero(); n]; n];
    for k in 0..n {
        for i in 0..n {
            let mut c = minor(t, i, k);
            if (i + k) % 2 == 1 {
                c = -c;
            }
            adj[k][i] = c;
        }
    }
    adj
}

/// The codifferent `𝔡⁻¹ = {x ∈ K : Tr(x·O_K) ⊆ ℤ}` (trace-dual fractional ideal):
/// `ℤ`-basis = the dual basis = columns of `T⁻¹ = (1/det)·adj(T)`.
pub fn codifferent(ord: &OrderData) -> Ideal {
    let n = ord.n;
    let t = trace_form(ord);
    let det = bareiss_det(&t);
    let adj = adjugate(&t);
    let cols: Vec<Vec<Integer>> =
        (0..n).map(|i| (0..n).map(|k| adj[k][i].clone()).collect()).collect();
    make_ideal(&cols, det.abs(), n)
}

/// The different `𝔡 = (𝔡⁻¹)⁻¹`, an integral ideal of `O_K` with `N(𝔡) = |disc_K|`.
pub fn different(ord: &OrderData) -> Ideal {
    crate::ideals::ideal_inverse(ord, &codifferent(ord))
}

/// The conductor `𝔣 = (O_K : ℤ[θ])` of the equation order in `O_K`, an integral
/// `O_K`-ideal with `N(𝔣) = [O_K : ℤ[θ]]² = disc(f)/disc_K`. Computed from the
/// suborder-different relation `(f'(θ)) = 𝔣·𝔡_K`, i.e. `𝔣 = (f'(θ))·𝔡_K⁻¹`.
pub fn conductor(f: &[Integer], ord: &OrderData) -> Ideal {
    let n = ord.n;
    // f'(θ) in power-basis coordinates, then to integral-basis coordinates
    let mut fprime = vec![Integer::zero(); n];
    for i in 1..f.len() {
        if i - 1 < n {
            fprime[i - 1] = f[i].clone() * Integer::from(i as i64);
        }
    }
    let fp = ord.power_to_order(&fprime);
    let fp_ideal = ideal_from_generators(ord, &[fp]); // (f'(θ))
    ideal_mul(ord, &fp_ideal, &codifferent(ord)) // (f'(θ))·𝔡_K⁻¹
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ideals::{ideal_norm, rational_prime_ideal};
    use crate::round2::{field_discriminant, maximal_order_data};

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    fn check(f: &[Integer], disc: i64) {
        let ord = maximal_order_data(f);
        // det(trace form) = disc_K (signed); cross-checks field_discriminant
        assert_eq!(bareiss_det(&trace_form(&ord)), Integer::from(disc));
        assert_eq!(field_discriminant(f), Integer::from(disc));
        // different is integral with N(𝔡) = |disc_K|
        let d = different(&ord);
        assert!(d.denom.is_one(), "different is integral");
        assert_eq!(ideal_norm(&d), Integer::from(disc.abs()));
    }

    #[test]
    fn different_norm_is_abs_disc() {
        check(&iz(&[1, 0, 1]), -4); // Q(i)
        check(&iz(&[5, 0, 1]), -20); // Q(sqrt(-5))
        check(&iz(&[-5, 0, 1]), 5); // Q(sqrt 5)
        check(&iz(&[-8, -2, -1, 1]), -503); // Dedekind's non-monogenic cubic
    }

    #[test]
    fn different_is_f_prime_for_monogenic() {
        // Q(i): O_K = Z[i], 𝔡 = (f'(θ)) = (2i) = (2).
        let f = iz(&[1, 0, 1]);
        let ord = maximal_order_data(&f);
        assert_eq!(different(&ord), rational_prime_ideal(&ord, 2));
    }

    #[test]
    fn conductor_norm_is_index_squared() {
        use crate::ideals::ideal_norm;
        // N(𝔣) = [O_K:Z[θ]]^2 = disc(f)/disc_K.
        let check = |f: &[Integer], idx2: i64| {
            let ord = maximal_order_data(f);
            let c = conductor(f, &ord);
            assert!(c.denom.is_one());
            assert_eq!(ideal_norm(&c), Integer::from(idx2));
        };
        check(&iz(&[1, 0, 1]), 1); // Q(i): index 1
        check(&iz(&[23, 0, 1]), 4); // x^2+23: disc -92, disc_K -23, index 2 → 4
        check(&iz(&[-8, -2, -1, 1]), 4); // Dedekind cubic: index 2 → 4
    }
}
