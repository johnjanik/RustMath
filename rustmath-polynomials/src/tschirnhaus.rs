//! Tschirnhaus transformations — relabelling the roots of a polynomial without
//! changing its splitting field, the keystone that makes resolvents *separable*.
//!
//! Let `f ∈ ℤ[x]` be monic and separable of degree `n` with roots `α₁,…,αₙ`, and let
//! `g ∈ ℤ[x]` be any polynomial. The **Tschirnhaus transform** of `f` by `g` is the
//! characteristic polynomial of the multiplication-by-`g(α)` map,
//! ```text
//!     T_g(Y) = ∏_{i=1}^{n} (Y − g(αᵢ)) = Res_x( f(x), Y − g(x) ) ∈ ℤ[Y],
//! ```
//! a monic integer polynomial of degree `n`. The roots `βᵢ = g(αᵢ)` are algebraic
//! integers lying in the same splitting field, and the Galois group acts on
//! `{β₁,…,βₙ}` by the *same* permutations it induces on `{α₁,…,αₙ}`
//! (`σ(βᵢ) = σ(g(αᵢ)) = g(σαᵢ)`). Hence whenever the `βᵢ` are **distinct**
//! (`T_g` separable) the relabelling is faithful: every group-theoretic invariant —
//! cycle types, `k`-subset orbit lengths — is preserved.
//!
//! ## Why it matters for the Stauduhar method
//!
//! The `k`-subset-sum resolvent `R(Y) = ∏_{|S|=k}(Y − Σ_{i∈S} αᵢ)` reads off the orbit
//! lengths of `Gal(f)` on `k`-subsets *only when its roots (the subset sums) are
//! distinct* — i.e. when `R` is separable. For special polynomials two different
//! subsets can share a sum (`α₁+α₂ = α₃+α₄`), collapsing `R` and destroying the
//! bridge. Replacing `f` by a generic Tschirnhaus transform `T_g` spreads the roots so
//! the subset sums separate, **without changing the group** —
//! [`separable_subset_sum_resolvent`] searches a deterministic family of `g`'s until
//! the resolvent is squarefree, the standard fix used by every real Galois-group
//! engine (PARI's `nfgaloisconj`, Magma/`GaloisGroup`, the Fieker–Klüners descent).
//!
//! Coefficient convention: little-endian `Vec`, `c[i]` = coefficient of `xⁱ`.

use crate::bivariate;
use crate::resolvent::subset_sum_resolvent;
use crate::zx;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

/// Is `f ∈ ℤ[x]` separable (squarefree over ℚ̄)? True iff `deg gcd(f, f′) = 0`.
/// Constants and the zero polynomial are *not* separable here (`false`).
pub fn is_separable(f: &[Integer]) -> bool {
    let f = zx::trim(f);
    if zx::degree(&f) < 1 {
        return false;
    }
    let d = zx::derivative(&f);
    zx::degree(&zx::subresultant_gcd(&f, &d)) == 0
}

/// Convert a little-endian `ℚ` polynomial to `ℤ`, requiring every coefficient to be
/// integral (the Tschirnhaus transform of a monic integral `f` always is).
fn to_z(p: &[Rational]) -> Vec<Integer> {
    let v: Vec<Integer> = p
        .iter()
        .map(|c| {
            assert!(c.is_integer(), "Tschirnhaus transform has a non-integral coefficient");
            c.numerator().clone()
        })
        .collect();
    zx::trim(&v)
}

/// The **Tschirnhaus transform** `T_g(Y) = ∏_i (Y − g(αᵢ))` of a monic separable
/// `f ∈ ℤ[x]` by `g ∈ ℤ[x]`, returned little-endian over `ℤ`, monic of degree `n`.
///
/// Computed exactly as `Res_x(f(x), Y − g(x))` via the validated bivariate resultant.
/// `g` is first reduced modulo `f` (which leaves every `g(αᵢ)` unchanged since
/// `f(αᵢ) = 0`), keeping the resultant small. `f` must be monic; panics otherwise.
///
/// Note: `T_g` is separable **iff** the values `g(αᵢ)` are pairwise distinct, i.e. iff
/// `g` separates the roots of `f`. When `T_g` is separable it is a faithful relabelling
/// of `f` (same splitting field, same Galois action); when it is not, `g(α)` generates
/// a proper subfield and `T_g` is a perfect power of that subfield's defining polynomial.
pub fn tschirnhaus_transform(f: &[Integer], g: &[Integer]) -> Vec<Integer> {
    let f = zx::trim(f);
    let n = zx::degree(&f);
    assert!(n >= 1, "tschirnhaus_transform: f must be non-constant");
    assert!(
        f[f.len() - 1] == Integer::one(),
        "tschirnhaus_transform: f must be monic"
    );

    // Reduce g mod f. f monic ⇒ pseudo_rem is the ordinary remainder (no lc scaling).
    let gr = zx::trim(&zx::pseudo_rem(g, &f));

    // f(x) as a bivariate constant in Y: fbiv[i] = [f_i].
    let fbiv: Vec<Vec<Rational>> =
        f.iter().map(|c| vec![Rational::from_integer(c.clone())]).collect();

    // h(x, Y) = Y − g(x), X-major: coeff of xⁱ is the Y-polynomial below.
    //   xⁱ (i ≥ 1): constant −gᵢ ;  x⁰: −g₀ + Y.
    let dim = gr.len().max(1);
    let mut gbiv: Vec<Vec<Rational>> = (0..dim)
        .map(|i| {
            let gi = gr.get(i).cloned().unwrap_or_else(Integer::zero);
            vec![Rational::from_integer(-gi)]
        })
        .collect();
    gbiv[0].push(Rational::from_i64(1)); // + Y at the x⁰ slot

    // Res_x(f, Y − g) = ∏_i (Y − g(αᵢ)), integral and monic of degree n.
    let res = bivariate::resultant_in_t(&fbiv, &gbiv);
    let t = to_z(&res);
    debug_assert_eq!(zx::degree(&t), n, "Tschirnhaus transform changed the degree");
    t
}

/// A deterministic family of Tschirnhaus polynomials `g` to try, in increasing
/// "spread", all of degree `< n` (so `g(α)` stays a genuine element of `ℚ(α)` with a
/// degree-`n` characteristic polynomial). The first entry is the identity `g = x`
/// (which reproduces `f`), so callers that only need a transform when the trivial one
/// fails pay nothing extra in the common case.
fn candidate_transforms(n: usize) -> Vec<Vec<Integer>> {
    let mut out = vec![vec![Integer::zero(), Integer::one()]]; // g = x
    if n <= 2 {
        return out;
    }
    let degmax = n - 1;
    let mut pushed = 0usize;
    // g = x + c·x^d, sweeping the high coefficient c and the degree d. Keeping the
    // linear term makes successive g's genuinely different relabellings.
    'outer: for spread in 1..=6i64 {
        for d in 2..=degmax {
            for &lead in &[1i64, -1, 2, -2, 3, -3] {
                let c = lead * spread;
                let mut g = vec![Integer::zero(); d + 1];
                g[1] = Integer::one();
                g[d] = Integer::from(c);
                out.push(g);
                pushed += 1;
                if pushed >= 256 {
                    break 'outer;
                }
            }
        }
    }
    out
}

/// Build a **separable** `k`-subset-sum resolvent of `f` by Tschirnhaus relabelling.
///
/// Returns `(resolvent, g)` where `g` is a transform such that `T_g(f)` is separable
/// (a faithful relabelling) *and* the `k`-subset-sum resolvent of `T_g(f)` is itself
/// squarefree — so its irreducible-factor degrees are exactly the orbit lengths of
/// `Gal(f)` on `k`-subsets, with no accidental collisions. Tries `g = x` first
/// (the no-op), then the deterministic family of [`candidate_transforms`], up to
/// `max_tries`. Returns `None` if none in that range works (raise `max_tries`).
///
/// `f` must be monic and separable, `1 ≤ k ≤ n`. The returned resolvent has degree
/// `C(n, k)`; the caller is responsible for keeping `C(n,k)` within
/// [`subset_sum_resolvent`]'s exact-build limit.
pub fn separable_subset_sum_resolvent(
    f: &[Integer],
    k: usize,
    max_tries: usize,
) -> Option<(Vec<Integer>, Vec<Integer>)> {
    let n = zx::degree(f) as usize;
    assert!(k >= 1 && k <= n, "need 1 ≤ k ≤ n");
    for g in candidate_transforms(n).into_iter().take(max_tries) {
        let ft = tschirnhaus_transform(f, &g);
        // The relabelling must be faithful: g(αᵢ) distinct ⟺ T_g separable.
        if !is_separable(&ft) {
            continue;
        }
        let r = subset_sum_resolvent(&ft, k);
        if is_separable(&r) {
            return Some((r, g));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolvent::{resolvent_orbit_signature, subset_sum_resolvent};

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn identity_transform_reproduces_f() {
        // g = x  ⇒  T_g = f.
        let f = iz(&[-1, -1, 0, 0, 0, 1]); // x⁵ − x − 1
        assert_eq!(tschirnhaus_transform(&f, &iz(&[0, 1])), f);
    }

    #[test]
    fn shift_transform_quadratic() {
        // f = x² − 2 (roots ±√2), g = x + 1 ⇒ roots 1 ± √2 ⇒ Y² − 2Y − 1.
        let f = iz(&[-2, 0, 1]);
        let t = tschirnhaus_transform(&f, &iz(&[1, 1]));
        assert_eq!(t, iz(&[-1, -2, 1]));
    }

    #[test]
    fn square_transform_cubic() {
        // f = x³ − 2 (roots = cube roots of 2), g = x² ⇒ roots = cube roots of 4 ⇒ Y³ − 4.
        let f = iz(&[-2, 0, 0, 1]);
        let t = tschirnhaus_transform(&f, &iz(&[0, 0, 1]));
        assert_eq!(t, iz(&[-4, 0, 0, 1]));
    }

    #[test]
    fn transform_preserves_field_and_separability() {
        // A generic transform of a separable irreducible f is again separable of the
        // same degree (g(α) is a primitive element).
        let f = iz(&[1, 1, 0, 0, 1]); // x⁴ + x + 1, Galois S₄
        for g in candidate_transforms(4).into_iter().take(20) {
            let t = tschirnhaus_transform(&f, &g);
            assert_eq!(zx::degree(&t), 4);
            assert!(is_separable(&t), "transform by {g:?} not separable");
        }
    }

    #[test]
    fn constant_value_gives_perfect_power() {
        // g reduces to a constant mod f ⇒ all g(αᵢ) equal ⇒ T_g = (Y − c)ⁿ.
        // Take g = f + 7 (≡ 7 mod f). n = 3 ⇒ (Y − 7)³ = Y³ − 21Y² + 147Y − 343.
        let f = iz(&[-2, 0, 0, 1]); // x³ − 2
        let g = iz(&[7 - 2, 0, 0, 1]); // x³ + 5 ≡ 7 (mod x³ − 2)
        let t = tschirnhaus_transform(&f, &g);
        assert_eq!(t, iz(&[-343, 147, -21, 1]));
        assert!(!is_separable(&t));
    }

    #[test]
    fn separable_resolvent_recovers_orbit_signature() {
        // x⁴ + x + 1 (S₄): the k=2 resolvent is already separable (degree-6 irreducible).
        // The Tschirnhaus-guarded builder must return the same orbit signature [6].
        let f = iz(&[1, 1, 0, 0, 1]);
        let (r, _g) = separable_subset_sum_resolvent(&f, 2, 50).expect("found a separable resolvent");
        assert!(is_separable(&r));
        assert_eq!(resolvent_orbit_signature(&r).unwrap(), vec![6]);
    }

    #[test]
    fn separable_resolvent_fixes_a_collapsing_case() {
        // x⁴ − 2 has Galois D₄ with root block {α, −α}: the pair-sum resolvent has the
        // colliding sum α + (−α) = 0 appearing with multiplicity, so the *raw* k=2
        // resolvent is inseparable. The Tschirnhaus-guarded builder repairs it and the
        // resulting orbit signature must still sum to C(4,2) = 6 and have > 1 part
        // (D₄ is intransitive on pairs).
        let f = iz(&[-2, 0, 0, 0, 1]);
        let raw = subset_sum_resolvent(&f, 2);
        assert!(!is_separable(&raw), "x⁴−2 raw pair resolvent should collide");
        let (r, g) = separable_subset_sum_resolvent(&f, 2, 100).expect("repaired resolvent");
        assert!(is_separable(&r), "repaired resolvent must be separable");
        assert_ne!(g, iz(&[0, 1]), "the identity transform cannot fix a real collision");
        let sig = resolvent_orbit_signature(&r).unwrap();
        assert_eq!(sig.iter().sum::<usize>(), 6);
        assert!(sig.len() > 1, "D₄ acts intransitively on pairs: {sig:?}");
    }
}
