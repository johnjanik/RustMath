//! Kummer ray class fields via the finite `K(S,n)` space (note Alg 2,
//! `docs/algorithm_notes/abext_notes.md` §2).
//!
//! When `μ_n ⊂ K`, a cyclic degree-`n` extension `L/K` is `K(γ^{1/n})`. The
//! previous P2a approach searched `γ ∈ K^×` **by coefficients** — an unbounded
//! search that hung (`real_quadratic_rcf_qsqrt3`). Alg 2 replaces it with
//! enumeration of the **finite** group `K(S,n)`: every radical candidate comes
//! from a finite `S`-unit space, never an open element search.
//!
//! This module implements the `n = 2` case (`μ₂ = {±1} ⊂ K` always), which
//! covers every quadratic Kummer layer — including the **narrow class field of
//! a real quadratic**, the canonical case the old search hung on. The space
//! `K(S,2)` is the `𝔽₂`-span (mod squares) of the torsion (`−1`), the
//! fundamental units, and the `S`-prime generators (the `O_{K,S}^×/squares`
//! part; when `Cl_{K,S}[2] = 0` this is all of `K(S,2)`). For each candidate
//! `γ` we record the **real places where `γ < 0`** — exactly the infinite
//! places ramified in `K(√γ)/K`, which is the **signature lever** the IGP
//! construction needs. `L/K` is `x² − γ`.
//!
//! Scope: the infinite-place ramification (signs) and the tame finite conductor
//! (`v_𝔭(γ)` odd, `𝔭 ∤ 2`) are computed exactly here. The wild `2`-adic
//! conductor exponent is **not** computed in-repo (it needs local `2`-adic
//! Kummer-symbol membership) — flagged where relevant.

use crate::ideals::{ideal_valuation, Ideal};
use crate::round2::{maximal_order_data, OrderData};
use crate::s_unit_gens::{s_unit_generators, to_power_basis};
use rustmath_integers::Integer;

/// A `K(S,2)` element `γ` and the ramification of `K(√γ)/K`.
#[derive(Clone, Debug)]
pub struct KummerQuadratic {
    /// `γ ∈ K^×` in integral-basis coords; `L = K(√γ)`, relative poly `x² − γ`.
    pub gamma: Vec<Integer>,
    /// Indices into the ascending real embeddings where `γ < 0` — exactly the
    /// real places ramified in `K(√γ)/K` (the signature lever).
    pub ramified_real_places: Vec<usize>,
    /// Tame finite ramification: `S`-primes `𝔭 ∤ 2` with `v_𝔭(γ)` odd (each
    /// contributes conductor exponent 1). Stored as the index into `s_primes`.
    pub tame_ramified_primes: Vec<usize>,
}

/// The finite `K(S,2)` generating set (the `O_{K,S}^×/squares` part): torsion
/// (`−1`), fundamental units, then the `S`-prime generators. Each entry is an
/// element of `K` in integral-basis coords; `K(S,2)` is their `𝔽₂`-span.
pub fn k_s_2_space(f: &[Integer], s_primes: &[Ideal]) -> Option<Vec<Vec<Integer>>> {
    let su = s_unit_generators(f, s_primes)?;
    let mut gens = vec![su.torsion];
    gens.extend(su.free_gens);
    Some(gens)
}

/// Real embeddings of `K` (real roots of `f`), ascending — matching the
/// real-place indexing used by [`crate::rayclass`].
fn real_embeddings(f: &[Integer]) -> Vec<f64> {
    let n = f.len() - 1;
    if n == 0 {
        return vec![];
    }
    let c: Vec<f64> = f.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let lead = c[n];
    // Durand–Kerner for all complex roots.
    #[derive(Clone, Copy)]
    struct Z {
        re: f64,
        im: f64,
    }
    let mul = |a: Z, b: Z| Z {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    };
    let eval = |z: Z| {
        let mut acc = Z { re: 0.0, im: 0.0 };
        for k in (0..=n).rev() {
            acc = mul(acc, z);
            acc.re += c[k];
        }
        acc
    };
    let mut roots: Vec<Z> = (0..n)
        .map(|k| {
            let ang = 2.0 * std::f64::consts::PI * (k as f64) / (n as f64) + 0.4;
            Z {
                re: 1.3 * ang.cos(),
                im: 1.3 * ang.sin(),
            }
        })
        .collect();
    for _ in 0..200 {
        for i in 0..n {
            let fi = eval(roots[i]);
            let mut denom = Z { re: lead, im: 0.0 };
            for j in 0..n {
                if j != i {
                    let d = Z {
                        re: roots[i].re - roots[j].re,
                        im: roots[i].im - roots[j].im,
                    };
                    denom = mul(denom, d);
                }
            }
            let dnorm = denom.re * denom.re + denom.im * denom.im;
            if dnorm > 1e-300 {
                let delta = Z {
                    re: (fi.re * denom.re + fi.im * denom.im) / dnorm,
                    im: (fi.im * denom.re - fi.re * denom.im) / dnorm,
                };
                roots[i].re -= delta.re;
                roots[i].im -= delta.im;
            }
        }
    }
    let mut reals: Vec<f64> = roots
        .iter()
        .filter(|z| z.im.abs() < 1e-6)
        .map(|z| z.re)
        .collect();
    reals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    reals
}

/// Value of `γ` (integral-basis coords) at a real embedding `θ = r`.
fn eval_real(ord: &OrderData, gamma: &[Integer], r: f64) -> f64 {
    let pb = to_power_basis(ord, gamma); // γ = Σ pbᵢ·θ^i
    let mut acc = 0.0;
    let mut pw = 1.0;
    for ci in &pb {
        let num = ci.numerator().to_f64().unwrap_or(0.0);
        let den = ci.denominator().to_f64().unwrap_or(1.0);
        acc += (num / den) * pw;
        pw *= r;
    }
    acc
}

/// Enumerate the finite `K(S,2)` space: every nontrivial combination
/// `γ = ∏ genᵢ^{eᵢ}` (`eᵢ ∈ {0,1}`, not all zero) with its ramification. This
/// is **bounded** — `2^{#gens} − 1` candidates — which is the whole point of
/// Alg 2 (no open coefficient search).
pub fn kummer_quadratic_candidates(f: &[Integer], s_primes: &[Ideal]) -> Vec<KummerQuadratic> {
    let ord = maximal_order_data(f);
    let gens = match k_s_2_space(f, s_primes) {
        Some(g) => g,
        None => return vec![],
    };
    let reals = real_embeddings(f);
    let two = Integer::from(2);
    // which S-primes are odd (tame) — for the tame finite conductor rule.
    let odd_prime: Vec<bool> = s_primes
        .iter()
        .map(|p| {
            // 𝔭 ∤ 2 iff N(𝔭) is odd
            !(crate::ideals::ideal_norm(p) % two.clone()).is_zero()
        })
        .collect();
    let r = gens.len();
    let mut out = Vec::new();
    for mask in 1u32..(1u32 << r) {
        let sel: Vec<&Vec<Integer>> = (0..r).filter(|&i| (mask >> i) & 1 == 1).map(|i| &gens[i]).collect();
        let mut gamma = sel[0].clone();
        for g in &sel[1..] {
            gamma = ord.mul(&gamma, g);
        }
        let ramified_real_places: Vec<usize> = reals
            .iter()
            .enumerate()
            .filter(|(_, &rr)| eval_real(&ord, &gamma, rr) < 0.0)
            .map(|(i, _)| i)
            .collect();
        let tame_ramified_primes: Vec<usize> = s_primes
            .iter()
            .enumerate()
            .filter(|(i, p)| odd_prime[*i] && ideal_valuation(&ord, &principal_ideal(&ord, &gamma), p) % 2 == 1)
            .map(|(i, _)| i)
            .collect();
        out.push(KummerQuadratic {
            gamma,
            ramified_real_places,
            tame_ramified_primes,
        });
    }
    out
}

/// The principal ideal `(γ)` (for the tame-valuation conductor check).
fn principal_ideal(ord: &OrderData, gamma: &[Integer]) -> Ideal {
    crate::ideals::ideal_from_generators(ord, &[gamma.to_vec()])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ideals::prime_ideals;
    use rustmath_rationals::Rational;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    fn is_power_basis_minus_one(ord: &OrderData, gamma: &[Integer]) -> bool {
        let pb = to_power_basis(ord, gamma);
        !pb.is_empty()
            && pb[0] == Rational::from_integer(-1)
            && pb.iter().skip(1).all(|x| *x == Rational::from_integer(0))
    }

    #[test]
    fn narrow_hcf_qsqrt3_kummer_generator_is_minus_one() {
        // Q(√3): gp gives narrow class field = x²+1 = K(√−1), rnfdisc 1
        // (ramified only at the two infinite places). Alg 2's finite K(S,2)
        // contains the torsion −1, which is totally negative ⇒ ramified at BOTH
        // real places = the narrow modulus. (The old prime-only kummer_generator
        // misses this unit/torsion generator entirely.)
        let f = iz(&[-3, 0, 1]);
        let (ord, p2v) = prime_ideals(&f, 2);
        let s_primes: Vec<Ideal> = p2v.into_iter().map(|(p, _, _)| p).collect();

        let cands = kummer_quadratic_candidates(&f, &s_primes);
        let minus_one = cands
            .iter()
            .find(|c| is_power_basis_minus_one(&ord, &c.gamma))
            .expect("−1 must be in the finite K(S,2) space");
        assert_eq!(
            minus_one.ramified_real_places,
            vec![0, 1],
            "γ=−1 is totally negative ⇒ both real places ramified (narrow modulus); \
             L = K(√−1), relative poly x²+1 (matches gp bnrclassfield)"
        );
        // −1 is a unit ⇒ no tame finite ramification.
        assert!(minus_one.tame_ramified_primes.is_empty());
    }

    #[test]
    fn fundamental_unit_qsqrt3_is_totally_positive() {
        // Contrast: the fundamental unit 2+√3 (≈3.73) has conjugate 2−√3 (≈0.27),
        // both positive ⇒ totally positive ⇒ K(√ε)/K unramified at ∞.
        let f = iz(&[-3, 0, 1]);
        let (ord, p2v) = prime_ideals(&f, 2);
        let s_primes: Vec<Ideal> = p2v.into_iter().map(|(p, _, _)| p).collect();
        let cands = kummer_quadratic_candidates(&f, &s_primes);
        // find a candidate evaluating to a totally positive non-rational element
        let eps = cands.iter().find(|c| {
            let pb = to_power_basis(&ord, &c.gamma);
            // has a θ component (not ±1) and is totally positive
            pb.len() > 1
                && pb[1] != Rational::from_integer(0)
                && c.ramified_real_places.is_empty()
        });
        assert!(
            eps.is_some(),
            "the totally-positive fundamental unit must appear in K(S,2)"
        );
    }

    #[test]
    fn k_s_2_space_is_finite_and_bounded() {
        // The space is a finite F₂-generating set — the structural anti-hang
        // property. For Q(√3) with S={𝔭₂}: torsion + fundamental unit +
        // 𝔭₂-generator ⇒ a small finite set, and 2^#gens − 1 candidates.
        let f = iz(&[-3, 0, 1]);
        let (_o, p2v) = prime_ideals(&f, 2);
        let s_primes: Vec<Ideal> = p2v.into_iter().map(|(p, _, _)| p).collect();
        let gens = k_s_2_space(&f, &s_primes).expect("space");
        assert!(gens.len() >= 2, "at least torsion + a unit");
        let cands = kummer_quadratic_candidates(&f, &s_primes);
        assert_eq!(cands.len(), (1usize << gens.len()) - 1);
    }
}
