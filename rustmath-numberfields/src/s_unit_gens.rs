//! Explicit GENERATORS of the S-unit group `O_{K,S}^×` of a number field
//! `K = ℚ[x]/(f)` (Module 8, WP-SUNIT-GEN).
//!
//! [`s_units`](crate::s_units) only computes the S-unit *rank* and the S-class
//! number. This module produces actual generators, as required by the Kummer
//! square-class constructor (WP-KUMMER): generators of `O_{K,S}^× / (O_{K,S}^×)²`
//! are obtained from these by reducing exponents mod 2.
//!
//! The clean structural decomposition (no Buchmann sub-exponential machinery
//! needed for small fields) is
//!
//! ```text
//!   O_{K,S}^×  =  ⟨−1⟩  ×  ⟨fundamental units of O_K⟩  ×  ⟨π_𝔭 : 𝔭 ∈ S⟩ ,
//! ```
//!
//! where `π_𝔭` generates `𝔭^{k}` for the least `k ≥ 1` with `𝔭^{k}` principal
//! (`k` divides `h_K`). The classes `[π_𝔭]` generate `O_{K,S}^× / O_K^×`, so
//! the union has the right free rank `r₁ + r₂ − 1 + |S|` (Dirichlet's S-unit
//! theorem) modulo torsion.
//!
//! ## Representation
//! Every K-element is returned in **integral-basis coordinates** (`Vec<Integer>`
//! of length `n`, the convention of [`crate::round2::OrderData`],
//! [`crate::ideals`], [`crate::classgroup`], and [`crate::units`]). A helper
//! [`to_power_basis`] converts to the power basis `1, θ, …, θ^{n−1}` as rational
//! coordinates for callers that want power-basis output.
//!
//! ## Scope / limitations
//! Fundamental units are found by a small-coefficient search over the maximal
//! order (same frontier as [`crate::units::regulator`]): exact for quadratic and
//! small cubic/quartic fields, but fields with a large regulator (a big
//! fundamental unit) are out of reach — [`fundamental_units`] then returns
//! `None`. The S-part needs `𝔭^k` principal for some small `k | h_K`; this uses
//! [`crate::classgroup::is_principal`] (an LLL short-vector search) and the
//! class group, so it inherits their field-size ceiling. `None` is returned
//! honestly whenever a piece cannot be certified rather than guessing.

use crate::classgroup::{class_group, element_norm, is_principal};
use crate::ideals::{ideal_mul, ideal_norm, one_ideal, Ideal};
use crate::round2::{maximal_order_data, OrderData};
use crate::units::signature;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

// --------------------------------------------------------------------------- //
// Output type
// --------------------------------------------------------------------------- //

/// Generators of `O_{K,S}^×` (modulo torsion), all in integral-basis coordinates.
///
/// The full group is `⟨torsion⟩ × ⟨g : g ∈ free_gens⟩`. `free_gens` has length
/// equal to the S-unit rank `r₁ + r₂ − 1 + |S|`: the first `unit_rank` entries
/// are fundamental units of `O_K`, the remaining `|S|` are the prime-power
/// generators `π_𝔭`.
#[derive(Debug, Clone)]
pub struct SUnitGenerators {
    /// The torsion generator (a root of unity generating `μ_K` up to the
    /// small-search resolution; always at least `−1`).
    pub torsion: Vec<Integer>,
    /// Free generators (units first, then S-prime generators), integral-basis
    /// coords. Length = [`crate::s_units::s_unit_rank`].
    pub free_gens: Vec<Vec<Integer>>,
    /// How many leading entries of `free_gens` are fundamental units of `O_K`
    /// (the rest are S-prime generators, in the same order as the input `S`).
    pub num_units: usize,
}

// --------------------------------------------------------------------------- //
// Minimal complex arithmetic + embeddings (mirrors units.rs / classgroup.rs)
// --------------------------------------------------------------------------- //

#[derive(Clone, Copy)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    fn mul(self, o: C) -> C {
        C { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
    fn add(self, o: C) -> C {
        C { re: self.re + o.re, im: self.im + o.im }
    }
    fn sub(self, o: C) -> C {
        C { re: self.re - o.re, im: self.im - o.im }
    }
    fn div(self, o: C) -> C {
        let d = o.re * o.re + o.im * o.im;
        C { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// Approximate complex roots of `f` (Durand–Kerner), as in `units.rs`.
fn roots(f: &[Integer]) -> Vec<C> {
    let n = f.len() - 1;
    let c: Vec<f64> = f.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
    let eval = |z: C| {
        let mut acc = C { re: 1.0, im: 0.0 };
        for k in (0..n).rev() {
            acc = acc.mul(z).add(C { re: c[k], im: 0.0 });
        }
        acc
    };
    let mut z: Vec<C> = (0..n)
        .map(|k| {
            let s = C { re: 0.4, im: 0.9 };
            let mut p = C { re: 1.0, im: 0.0 };
            for _ in 0..k {
                p = p.mul(s);
            }
            p
        })
        .collect();
    for _ in 0..300 {
        let mut md = 0.0f64;
        for i in 0..n {
            let mut den = C { re: 1.0, im: 0.0 };
            for j in 0..n {
                if j != i {
                    den = den.mul(z[i].sub(z[j]));
                }
            }
            let d = eval(z[i]).div(den);
            z[i] = z[i].sub(d);
            md = md.max(d.abs());
        }
        if md < 1e-13 {
            break;
        }
    }
    z
}

/// Real embeddings, then one representative per complex-conjugate pair.
fn embeddings(rts: &[C]) -> (Vec<C>, Vec<C>) {
    let mut reals = Vec::new();
    let mut cplx = Vec::new();
    let mut used = vec![false; rts.len()];
    for i in 0..rts.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        if rts[i].im.abs() < 1e-6 {
            reals.push(rts[i]);
        } else {
            cplx.push(rts[i]);
            let mut best = usize::MAX;
            let mut bd = f64::INFINITY;
            for j in 0..rts.len() {
                if j != i && !used[j] {
                    let d = (rts[i].re - rts[j].re).abs() + (rts[i].im + rts[j].im).abs();
                    if d < bd {
                        bd = d;
                        best = j;
                    }
                }
            }
            if best != usize::MAX {
                used[best] = true;
            }
        }
    }
    (reals, cplx)
}

/// `Log(α) = (eᵢ·log|σᵢ(α)|)` of length `r₁+r₂` for `α` in integral-basis coords.
fn log_embedding(ord: &OrderData, alpha: &[Integer], reals: &[C], cplx: &[C]) -> Vec<f64> {
    let n = ord.n;
    let dd = ord.d.to_f64().unwrap_or(1.0);
    let pow: Vec<f64> = (0..n)
        .map(|i| {
            let mut s = 0.0f64;
            for k in 0..n {
                s += ord.w[i][k].to_f64().unwrap_or(0.0) * alpha[k].to_f64().unwrap_or(0.0);
            }
            s
        })
        .collect();
    let sigma = |r: C| -> C {
        let mut acc = C { re: 0.0, im: 0.0 };
        let mut pw = C { re: 1.0, im: 0.0 };
        for &coef in &pow {
            acc = acc.add(C { re: coef, im: 0.0 }.mul(pw));
            pw = pw.mul(r);
        }
        C { re: acc.re / dd, im: acc.im / dd }
    };
    let mut out = Vec::with_capacity(reals.len() + cplx.len());
    for r in reals {
        out.push(sigma(*r).abs().ln());
    }
    for r in cplx {
        out.push(2.0 * sigma(*r).abs().ln());
    }
    out
}

/// Enumerate units of `O_K` with integral-basis coordinates in `[−b, b]ⁿ`
/// (`|N(α)| = 1`), as `(coords, Log(α))`.
fn small_units(ord: &OrderData, reals: &[C], cplx: &[C], b: i64) -> Vec<(Vec<Integer>, Vec<f64>)> {
    let n = ord.n;
    let mut out = Vec::new();
    let mut idx = vec![-b; n];
    loop {
        let alpha: Vec<Integer> = idx.iter().map(|&x| Integer::from(x)).collect();
        if alpha.iter().any(|x| !x.is_zero()) && element_norm(ord, &alpha).abs().is_one() {
            let lv = log_embedding(ord, &alpha, reals, cplx);
            out.push((alpha, lv));
        }
        let mut p = 0;
        while p < n {
            idx[p] += 1;
            if idx[p] > b {
                idx[p] = -b;
                p += 1;
            } else {
                break;
            }
        }
        if p == n {
            break;
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// Fundamental units
// --------------------------------------------------------------------------- //

/// A fundamental system of units of `O_K`: `unit_rank(f)` multiplicatively
/// independent units (independent log-embedding vectors), in integral-basis
/// coordinates. Returns `Some(vec![])` for unit rank 0 (imaginary quadratic / ℚ).
///
/// Found by a small-coefficient search over `O_K` (the frontier of
/// [`crate::units::regulator`]); `None` if no full independent system is found
/// within the search bound (large regulator).
pub fn fundamental_units(f: &[Integer]) -> Option<Vec<Vec<Integer>>> {
    let ord = maximal_order_data(f);
    fundamental_units_with(f, &ord)
}

/// Like [`fundamental_units`] but reusing a precomputed order.
fn fundamental_units_with(f: &[Integer], ord: &OrderData) -> Option<Vec<Vec<Integer>>> {
    let (r1, r2) = signature(f);
    let rank = r1 + r2 - 1;
    if rank == 0 {
        return Some(Vec::new());
    }
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    let b = if f.len() - 1 <= 2 { 40 } else { 6 };
    let units = small_units(ord, &reals, &cplx, b);

    // candidate (coords, projected log vector) — drop one coord (trace-zero) and
    // discard torsion (all-zero log).
    let mut cands: Vec<(Vec<Integer>, Vec<f64>)> = units
        .into_iter()
        .filter(|(_a, lv)| lv.iter().any(|x| x.abs() >= 1e-7))
        .map(|(a, lv)| (a, lv[..rank].to_vec()))
        .collect();
    // shortest log vectors first (a fundamental system for small fields)
    cands.sort_by(|a, b| {
        let na: f64 = a.1.iter().map(|x| x * x).sum();
        let nb: f64 = b.1.iter().map(|x| x * x).sum();
        na.partial_cmp(&nb).unwrap()
    });

    // greedily pick `rank` log-independent units (Gram–Schmidt independence test)
    let mut chosen: Vec<Vec<Integer>> = Vec::new();
    let mut ortho: Vec<Vec<f64>> = Vec::new();
    for (coords, v) in cands {
        let mut r = v.clone();
        for o in &ortho {
            let dot: f64 = r.iter().zip(o).map(|(a, b)| a * b).sum();
            let nn: f64 = o.iter().map(|x| x * x).sum();
            if nn > 1e-18 {
                for i in 0..r.len() {
                    r[i] -= dot / nn * o[i];
                }
            }
        }
        if r.iter().map(|x| x * x).sum::<f64>().sqrt() > 1e-6 {
            ortho.push(r);
            chosen.push(coords);
            if chosen.len() == rank {
                break;
            }
        }
    }
    if chosen.len() < rank {
        return None;
    }
    Some(chosen)
}

/// The torsion generator: a root of unity generating `μ_K` to the resolution of
/// the small search (returns a primitive one when found, else `−1`).
fn torsion_generator(f: &[Integer], ord: &OrderData) -> Vec<Integer> {
    let neg_one: Vec<Integer> = ord.one().iter().map(|c| -c.clone()).collect();
    let rts = roots(f);
    let (reals, cplx) = embeddings(&rts);
    // roots of unity = units with |σ(α)| = 1 at every embedding (all-zero log).
    let b = if f.len() - 1 <= 2 { 6 } else { 4 };
    let units = small_units(ord, &reals, &cplx, b);
    // pick the torsion element of largest multiplicative order; approximate the
    // order by the smallest k with α^k == 1. Default to −1.
    let mut best = neg_one.clone();
    let mut best_order = 2usize;
    for (a, lv) in &units {
        if lv.iter().all(|x| x.abs() < 1e-7) {
            let ord_a = torsion_order(ord, a);
            if ord_a > best_order {
                best_order = ord_a;
                best = a.clone();
            }
        }
    }
    best
}

/// Smallest `k ≥ 1` with `α^k = 1` in `O_K`, capped (torsion is small). `1` if
/// `α = 1`, `0` (treated as "not torsion") if no small power is `1`.
fn torsion_order(ord: &OrderData, a: &[Integer]) -> usize {
    let one = ord.one();
    let mut cur = a.to_vec();
    for k in 1..=24usize {
        if cur == one {
            return k;
        }
        cur = ord.mul(&cur, a);
    }
    0
}

// --------------------------------------------------------------------------- //
// S-part generators
// --------------------------------------------------------------------------- //

/// `𝔞^e` by binary exponentiation.
fn ideal_pow(ord: &OrderData, a: &Ideal, e: usize) -> Ideal {
    let mut result = one_ideal(ord);
    let mut base = a.clone();
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            result = ideal_mul(ord, &result, &base);
        }
        e >>= 1;
        if e > 0 {
            base = ideal_mul(ord, &base, &base);
        }
    }
    result
}

/// For each prime `𝔭 ∈ s_primes`, a generator `π_𝔭` of `𝔭^{k_𝔭}` with `k_𝔭`
/// the least `k ≥ 1` (dividing `h_K`) for which `𝔭^k` is principal. These
/// generate `O_{K,S}^× / O_K^×`. Integral-basis coords, in the input order.
///
/// `None` if the class group is unavailable or no principal power is found
/// within `k ≤ h_K` (should not happen, since `𝔭^{h_K}` is always principal,
/// but [`is_principal`] is a heuristic short-vector search).
pub fn s_part_generators(f: &[Integer], s_primes: &[Ideal]) -> Option<Vec<Vec<Integer>>> {
    let ord = maximal_order_data(f);
    s_part_generators_with(f, &ord, s_primes)
}

fn s_part_generators_with(
    f: &[Integer],
    ord: &OrderData,
    s_primes: &[Ideal],
) -> Option<Vec<Vec<Integer>>> {
    if s_primes.is_empty() {
        return Some(Vec::new());
    }
    let cg = class_group(f)?;
    let h: usize = cg.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(s_primes.len());
    for p in s_primes {
        let mut found = None;
        // k | h_K, and 𝔭^h is always principal; try every k up to h.
        for k in 1..=h {
            let pk = ideal_pow(ord, p, k);
            if let Some(gen) = is_principal(f, ord, &pk) {
                found = Some(gen);
                break;
            }
        }
        out.push(found?);
    }
    Some(out)
}

// --------------------------------------------------------------------------- //
// Full S-unit generators
// --------------------------------------------------------------------------- //

/// Explicit generators of `O_{K,S}^×` (modulo torsion) for `K = ℚ[x]/(f)` and
/// the set `S = s_primes` of prime ideals.
///
/// Returns [`SUnitGenerators`] whose `free_gens` has length equal to the S-unit
/// rank `r₁ + r₂ − 1 + |S|` ([`crate::s_units::s_unit_rank`]): fundamental units
/// of `O_K` first, then the S-prime generators `π_𝔭`. `None` if either piece
/// cannot be certified (large regulator, or class-group/principality search
/// failed).
pub fn s_unit_generators(f: &[Integer], s_primes: &[Ideal]) -> Option<SUnitGenerators> {
    let ord = maximal_order_data(f);
    let units = fundamental_units_with(f, &ord)?;
    let s_part = s_part_generators_with(f, &ord, s_primes)?;
    let torsion = torsion_generator(f, &ord);
    let num_units = units.len();
    let mut free_gens = units;
    free_gens.extend(s_part);
    Some(SUnitGenerators { torsion, free_gens, num_units })
}

// --------------------------------------------------------------------------- //
// Power-basis conversion + S-unit verification helpers
// --------------------------------------------------------------------------- //

/// Convert an element from integral-basis coordinates to power-basis rational
/// coordinates `(c₀, …, c_{n−1})` so that the element is `Σ cᵢ θ^i`.
///
/// In the maximal order, basis element `j` is `(1/d)·Σ_i w[i][j]·θ^i`, so for
/// integral coords `α = Σ_j αⱼ·(basis j)` the power-basis numerator at `θ^i` is
/// `Σ_j w[i][j]·αⱼ`, all over the denominator `d`.
pub fn to_power_basis(ord: &OrderData, alpha: &[Integer]) -> Vec<Rational> {
    let n = ord.n;
    (0..n)
        .map(|i| {
            let mut num = Integer::zero();
            for j in 0..n {
                num = num + ord.w[i][j].clone() * alpha[j].clone();
            }
            Rational::new(num, ord.d.clone()).expect("nonzero denom")
        })
        .collect()
}

/// `true` if `α` (integral-basis coords) is a genuine S-unit: its norm — hence
/// its principal ideal `(α)` — is supported entirely on rational primes lying
/// under `S` (i.e. `|N(α)|` is `S`-smooth). For a unit, `|N(α)| = 1` (vacuously
/// S-smooth). The rational primes under `S` are those dividing the `N(𝔭)`.
pub fn is_s_unit(ord: &OrderData, alpha: &[Integer], s_primes: &[Ideal]) -> bool {
    let mut nrm = element_norm(ord, alpha).abs();
    if nrm.is_zero() {
        return false;
    }
    // collect the rational primes below S (prime factors of each N(𝔭))
    let mut s_rat: Vec<Integer> = Vec::new();
    for p in s_primes {
        let np = ideal_norm(p);
        for q in prime_factors(&np) {
            if !s_rat.contains(&q) {
                s_rat.push(q);
            }
        }
    }
    // strip all S-rational-prime factors from |N(α)|
    for q in &s_rat {
        while (nrm.clone() % q.clone()).is_zero() {
            nrm = nrm / q.clone();
        }
    }
    nrm.is_one()
}

/// Distinct rational prime factors of a positive integer (trial division; the
/// leftover cofactor taken as one prime).
fn prime_factors(n: &Integer) -> Vec<Integer> {
    let mut m = n.abs();
    let mut out = Vec::new();
    let mut d = Integer::from(2);
    while d.clone() * d.clone() <= m {
        if (m.clone() % d.clone()).is_zero() {
            out.push(d.clone());
            while (m.clone() % d.clone()).is_zero() {
                m = m / d.clone();
            }
        }
        d = d + Integer::one();
    }
    if m > Integer::one() {
        out.push(m);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ideals::{factor_ideal, ideal_from_generators, prime_ideals};
    use crate::s_units::s_unit_rank;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    fn prime_above(f: &[Integer], p: i64) -> Ideal {
        prime_ideals(f, p).1.into_iter().next().unwrap().0
    }

    /// Number of independent free generators returned (sanity: equals rank).
    fn count_free(g: &SUnitGenerators) -> usize {
        g.free_gens.len()
    }

    // --- Fundamental units ------------------------------------------------- //

    #[test]
    fn fundamental_unit_real_quadratic() {
        // Q(√2): integral basis ω = (1, θ) with θ = √2 (disc 8, ℤ[√2] maximal).
        // Fundamental unit is 1 + √2 = coords [1, 1]; rank 1.
        let f = iz(&[-2, 0, 1]);
        let us = fundamental_units(&f).expect("rank-1 fundamental unit");
        assert_eq!(us.len(), 1);
        let ord = maximal_order_data(&f);
        // it must be a unit (|N| = 1) and not torsion
        let u = &us[0];
        assert!(element_norm(&ord, u).abs().is_one(), "fundamental element is a unit");
        assert_ne!(torsion_order(&ord, u), 1, "not the trivial element");
        // N(1+√2) = 1·1 − 2·1 = −1, |N| = 1 — check the actual found unit's norm.
        // (The search may return ±the fundamental unit or its inverse, all valid.)
    }

    #[test]
    fn fundamental_units_imaginary_quadratic_empty() {
        // Q(√−5): unit rank 0 ⇒ no fundamental units.
        let f = iz(&[5, 0, 1]);
        assert_eq!(fundamental_units(&f).unwrap().len(), 0);
        // Q(i): rank 0 too.
        assert_eq!(fundamental_units(&iz(&[1, 0, 1])).unwrap().len(), 0);
    }

    // --- S-part generators ------------------------------------------------- //

    #[test]
    fn s_part_principal_prime_is_self_generator() {
        // Q(√2): h = 1, so every prime is principal (k = 1). The prime above 7
        // (7 ≡ 1 mod 8 splits; 2 is a QR mod 7) yields a principal generator π
        // with (π) = 𝔭, i.e. |N(π)| = 7.
        let f = iz(&[-2, 0, 1]);
        let p7 = prime_above(&f, 7);
        let gens = s_part_generators(&f, std::slice::from_ref(&p7)).unwrap();
        assert_eq!(gens.len(), 1);
        let ord = maximal_order_data(&f);
        // (π) = 𝔭^1 ⇒ |N(π)| = N(𝔭) = 7
        assert_eq!(element_norm(&ord, &gens[0]).abs(), Integer::from(7));
        // and (π) really equals 𝔭
        let pi_ideal = ideal_from_generators(&ord, &[gens[0].clone()]);
        assert_eq!(pi_ideal, p7, "(π) = 𝔭");
    }

    #[test]
    fn s_part_nonprincipal_prime_uses_power() {
        // Q(√−5): h = 2. The prime 𝔭₂ above 2 is non-principal but 𝔭₂² = (2) is
        // principal ⇒ k = 2, generator π with (π) = 𝔭₂², |N(π)| = N(𝔭₂)² = 4.
        let f = iz(&[5, 0, 1]);
        let p2 = prime_above(&f, 2);
        let gens = s_part_generators(&f, std::slice::from_ref(&p2)).unwrap();
        assert_eq!(gens.len(), 1);
        let ord = maximal_order_data(&f);
        assert_eq!(element_norm(&ord, &gens[0]).abs(), Integer::from(4));
        // the generated ideal is 𝔭₂² and is S-supported (norm 4 = 2²)
        assert!(is_s_unit(&ord, &gens[0], std::slice::from_ref(&p2)));
    }

    // --- Full S-unit generators: rank cross-check -------------------------- //

    #[test]
    fn rank_crosscheck_real_quadratic_empty_s() {
        // Q(√2), S = ∅: generators = {fundamental unit}, count = rank = 1.
        let f = iz(&[-2, 0, 1]);
        let g = s_unit_generators(&f, &[]).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 0));
        assert_eq!(count_free(&g), 1);
        assert_eq!(g.num_units, 1);
    }

    #[test]
    fn rank_crosscheck_real_quadratic_with_s() {
        // Q(√2), S = {𝔭₇}: rank = unit_rank(1) + |S|(1) = 2.
        let f = iz(&[-2, 0, 1]);
        let p7 = prime_above(&f, 7);
        let g = s_unit_generators(&f, std::slice::from_ref(&p7)).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 1));
        assert_eq!(count_free(&g), 2);
        assert_eq!(g.num_units, 1); // 1 unit + 1 S-generator
    }

    #[test]
    fn rank_crosscheck_imaginary_quadratic_with_s() {
        // Q(√−5), S = {𝔭₂}: unit rank 0, so generators are exactly the S-prime
        // generators. rank = 0 + 1 = 1.
        let f = iz(&[5, 0, 1]);
        let p2 = prime_above(&f, 2);
        let g = s_unit_generators(&f, std::slice::from_ref(&p2)).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 1));
        assert_eq!(count_free(&g), 1);
        assert_eq!(g.num_units, 0);
        // each generator is a genuine S-unit
        for gen in &g.free_gens {
            let ord = maximal_order_data(&f);
            assert!(is_s_unit(&ord, gen, std::slice::from_ref(&p2)));
        }
    }

    #[test]
    fn rank_crosscheck_imaginary_quadratic_two_s_primes() {
        // Q(√−5), S = {𝔭₂, 𝔭₃} (3 splits since −5 ≡ 1 mod 3). rank = 0 + 2 = 2.
        let f = iz(&[5, 0, 1]);
        let p2 = prime_above(&f, 2);
        let p3 = prime_above(&f, 3);
        let s = vec![p2, p3];
        let g = s_unit_generators(&f, &s).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 2));
        assert_eq!(count_free(&g), 2);
        let ord = maximal_order_data(&f);
        for gen in &g.free_gens {
            assert!(is_s_unit(&ord, gen, &s), "every generator is an S-unit");
        }
    }

    // --- Each generator is a genuine S-unit (ideal S-supported) ------------ //

    #[test]
    fn generators_are_s_units_qi() {
        // Q(i), S = {𝔭₅}: unit rank 0 (torsion = i). One S-generator with
        // (π) = 𝔭₅ (5 splits, h = 1), |N(π)| = 5, S-smooth.
        let f = iz(&[1, 0, 1]);
        let p5 = prime_above(&f, 5);
        let g = s_unit_generators(&f, std::slice::from_ref(&p5)).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 1));
        assert_eq!(count_free(&g), 1);
        let ord = maximal_order_data(&f);
        // the S-generator's ideal factors over the prime above 5 only
        let pi_ideal = ideal_from_generators(&ord, &[g.free_gens[0].clone()]);
        let fac = factor_ideal(&f, &pi_ideal);
        // all primes in the factorization lie above 5
        for (pr, _v, _fdeg) in &fac {
            let np = ideal_norm(pr);
            assert!(
                prime_factors(&np).iter().all(|q| *q == Integer::from(5)),
                "S-generator supported only over 5"
            );
        }
        // torsion should have order > 2 in Q(i) (i has order 4); at minimum −1.
        assert!(torsion_order(&ord, &g.torsion) >= 2);
    }

    // --- Power-basis conversion -------------------------------------------- //

    #[test]
    fn power_basis_roundtrip_quadratic() {
        // Q(√2): ω = (1, θ). Coords [3, 5] ↦ 3 + 5θ in the power basis.
        let f = iz(&[-2, 0, 1]);
        let ord = maximal_order_data(&f);
        let pb = to_power_basis(&ord, &iz(&[3, 5]));
        assert_eq!(pb.len(), 2);
        assert_eq!(pb[0], Rational::from_integer(3));
        assert_eq!(pb[1], Rational::from_integer(5));
    }

    // --- Cubic: shows it scales beyond quadratics -------------------------- //

    #[test]
    fn cubic_fundamental_unit() {
        // x³ − x − 1: complex cubic, signature (1,1), unit rank 1. The regulator
        // is ~0.281 (units.rs validates it), so a fundamental unit exists in the
        // small search.
        let f = iz(&[-1, -1, 0, 1]);
        let us = fundamental_units(&f).expect("cubic fundamental unit");
        assert_eq!(us.len(), 1);
        let ord = maximal_order_data(&f);
        assert!(element_norm(&ord, &us[0]).abs().is_one());
    }

    #[test]
    fn cubic_s_unit_generators_rank() {
        // x³ − x − 1 has class number 1, so any prime is principal (k=1).
        // S = {𝔭₅} (some prime above 5). rank = unit_rank(1) + 1 = 2.
        let f = iz(&[-1, -1, 0, 1]);
        let p5 = prime_above(&f, 5);
        let g = s_unit_generators(&f, std::slice::from_ref(&p5)).unwrap();
        assert_eq!(count_free(&g), s_unit_rank(&f, 1));
        assert_eq!(count_free(&g), 2);
        assert_eq!(g.num_units, 1);
        let ord = maximal_order_data(&f);
        assert!(is_s_unit(&ord, &g.free_gens[1], std::slice::from_ref(&p5)));
    }
}
