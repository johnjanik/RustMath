//! Numeric evaluation of **relative resolvents** at the labeled complex roots,
//! plus the rational-root (Stauduhar) criterion.
//!
//! # The relative resolvent
//!
//! Let `f ∈ ℤ[x]` be monic, separable, of degree `n`, with complex roots
//! `α_0, …, α_{n−1}` carrying the *stable labeling* from
//! [`rustmath_polynomials::root_label::complex_roots`]. Let `G ⊆ S_n` be the
//! current group in the descent and `H < G` a maximal subgroup. We need a
//! polynomial invariant `F` with `Stab_G(F) = H`; the relative resolvent is
//! ```text
//!     R_{G,H}(Y) = ∏_{c ∈ G/H} (Y − F(c · α)),     deg R = [G : H].
//! ```
//! Because `F` is `H`-invariant, `F(c·α)` depends only on the coset `cH`, so the
//! product is over coset representatives. The coefficients of `R` are symmetric
//! under `G`; hence when `Gal(f) ⊆ G` they are *fixed by `Gal(f)`* and therefore
//! lie in `ℚ` (in `ℤ` for monic integral `f`).
//!
//! **Stauduhar's criterion.** If `R_{G,H}` has a *simple rational root*
//! `θ = F(c·α)`, then `Gal(f)` fixes that root, i.e. `Gal(f) ⊆ c H c⁻¹` — a
//! conjugate of `H`. We then descend into that conjugate.
//!
//! # The invariant `F`
//!
//! We use a **generic symmetrized linear form**: with an integer weight vector
//! `β = (β_0,…,β_{n−1})` set `L(x) = Σ_i β_i x_i` and
//! ```text
//!     F(α) = Σ_{h ∈ H} L(h · α)^d        (d ≥ 2).
//! ```
//! `F` is manifestly invariant under the left action of `H`. For *generic* `β`
//! (and a high-enough power `d`) the only elements of `G` fixing `F` are exactly
//! those of `H`, so the `[G:H]` coset values `θ_c = F(c·α)` are pairwise
//! distinct and `R` is separable. We verify the distinctness numerically and, on
//! a (rare) collision, re-pick `β` / raise `d` ([`build_relative_resolvent`]).
//!
//! All evaluation is done in arbitrary precision via
//! [`rustmath_polynomials::root_label::BigComplex`]; the rational-root test uses
//! [`rustmath_polynomials::root_label::complex_round_to_integer_if_close`] and
//! [`rustmath_polynomials::root_label::rational_reconstruction`] with a tolerance
//! comfortably below the working precision.

use crate::perm::{compose, Perm};
use rustmath_integers::Integer;
use rustmath_polynomials::root_label::{
    complex_round_to_integer_if_close, rational_reconstruction, BigComplex, BigFloat,
};
use rustmath_rationals::Rational;

/// One coset value `θ_c = F(c·α)` together with the coset representative `c`
/// that produced it (needed to identify the conjugate of `H` to descend into).
#[derive(Clone)]
pub struct CosetValue {
    /// The coset representative `c` (a permutation of `{0,…,n−1}`).
    pub rep: Perm,
    /// The numeric value `θ_c = F(c·α)`.
    pub value: BigComplex,
}

/// A built relative resolvent: the coset values and the working precision.
pub struct RelativeResolvent {
    /// One value per coset of `G/H`, in coset-rep order.
    pub coset_values: Vec<CosetValue>,
    /// Working precision (fractional bits) the values were computed at.
    pub prec: u32,
}

/// Evaluate the generic linear form `L(α') = Σ_i β_i α'_i`, where `α'_i = α[p[i]]`
/// is the root reindexed by the permutation `p` (the left action `p·α`).
fn eval_linear_form(beta: &[Integer], roots: &[BigComplex], p: &Perm, prec: u32) -> BigComplex {
    let mut acc = BigComplex::zero(prec);
    for (i, &pi) in p.iter().enumerate() {
        // term = β_i · α_{p[i]}
        let coeff = BigComplex::new(BigFloat::from_integer(&beta[i], prec), BigFloat::zero(prec));
        acc = acc.add(&coeff.mul(&roots[pi]));
    }
    acc
}

/// Raise a [`BigComplex`] to a small non-negative integer power.
fn cpow(z: &BigComplex, d: u32) -> BigComplex {
    let mut acc = BigComplex::new(BigFloat::from_i64(1, z.prec()), BigFloat::zero(z.prec()));
    for _ in 0..d {
        acc = acc.mul(z);
    }
    acc
}

/// The invariant value `F(c·α) = Σ_{h∈H} L((c∘h)·α)^d`.
///
/// Here `H` is given by its explicit element list `h_elems`, `c` is the coset
/// representative, `beta` the weight vector, and `d` the power.
fn eval_invariant(
    beta: &[Integer],
    roots: &[BigComplex],
    h_elems: &[Perm],
    c: &Perm,
    d: u32,
    prec: u32,
) -> BigComplex {
    let mut acc = BigComplex::zero(prec);
    for h in h_elems {
        let ch = compose(c, h); // (c∘h) applied to indices
        let l = eval_linear_form(beta, roots, &ch, prec);
        acc = acc.add(&cpow(&l, d));
    }
    acc
}

/// Build the relative resolvent `R_{G,H}` numerically: evaluate `θ_c = F(c·α)`
/// for every coset representative `c ∈ reps`, choosing the weight vector `beta`
/// and power `d`. Returns `None` if two coset values coincide to within the
/// distinctness tolerance (caller should re-pick `beta` / raise `d`).
///
/// `tol_bits` is the resolution (in bits) at which two coset values are
/// considered equal; it must be well below `prec`.
pub fn build_relative_resolvent(
    beta: &[Integer],
    roots: &[BigComplex],
    h_elems: &[Perm],
    reps: &[Perm],
    d: u32,
    prec: u32,
    tol_bits: u32,
) -> Option<RelativeResolvent> {
    let mut coset_values: Vec<CosetValue> = Vec::with_capacity(reps.len());
    for c in reps {
        let value = eval_invariant(beta, roots, h_elems, c, d, prec);
        coset_values.push(CosetValue { rep: c.clone(), value });
    }
    // Distinctness check: all pairwise differences must exceed the tolerance.
    for i in 0..coset_values.len() {
        for j in (i + 1)..coset_values.len() {
            let diff = coset_values[i].value.sub(&coset_values[j].value);
            if is_near_zero(&diff, tol_bits) {
                return None; // collision → resolvent not separable with this β,d
            }
        }
    }
    Some(RelativeResolvent { coset_values, prec })
}

/// True if a [`BigComplex`] is within `2^{−tol_bits}` of zero in both parts.
fn is_near_zero(z: &BigComplex, tol_bits: u32) -> bool {
    near_zero_bf(&z.re, tol_bits) && near_zero_bf(&z.im, tol_bits)
}

fn near_zero_bf(x: &BigFloat, tol_bits: u32) -> bool {
    // |x| ≤ 2^{−tol_bits}  ⟺  |mantissa| ≤ 2^{prec − tol_bits}.
    let prec = x.prec();
    let bound = if tol_bits <= prec {
        // 2^{prec − tol_bits}
        Integer::from(2i64).pow(prec - tol_bits)
    } else {
        Integer::from(0i64)
    };
    x.mantissa().abs() <= bound
}

/// The Stauduhar rational-root outcome of a relative resolvent.
pub struct RationalRoot {
    /// The coset representative `c` whose value `θ_c` is rational.
    pub rep: Perm,
    /// The rational value `θ_c` (an integer is returned as `p/1`).
    pub value: Rational,
}

/// Apply **Stauduhar's criterion**: scan the (already-separable) resolvent's
/// coset values for a *rational* root and return the first coset that attains
/// one. Because [`build_relative_resolvent`] only returns a resolvent whose
/// coset values are pairwise distinct (to tolerance), every rational coset value
/// is automatically a *simple* root of `R_{G,H}` — which is exactly Stauduhar's
/// requirement. A rational value at coset `c` certifies `Gal(f) ⊆ c·H·c⁻¹`, so
/// any rational coset is a sound, progress-making descent target.
///
/// Note: when `H` is *normal* in `G` (e.g. `V₄ ◁ A₄`) and `Gal = H`, **every**
/// coset value is rational and they all point to the same conjugate `H`; we
/// simply take the first. When `Gal ⊄ H` the separable resolvent has *no*
/// rational root and this returns `None` — the definitive negative.
///
/// `tol_bits` must be comfortably below `prec`.
pub fn find_simple_rational_root(
    res: &RelativeResolvent,
    max_denom: &Integer,
    tol_bits: u32,
) -> Option<RationalRoot> {
    for cv in res.coset_values.iter() {
        if let Some(q) = value_to_rational(&cv.value, max_denom, tol_bits) {
            return Some(RationalRoot { rep: cv.rep.clone(), value: q });
        }
    }
    None
}

/// Recognise a [`BigComplex`] coset value as a rational: imaginary part ≈ 0 and
/// real part ≈ a small-denominator rational. Integers are caught first (the
/// common case for monic integral `f`), then general rationals via continued
/// fractions.
pub fn value_to_rational(
    z: &BigComplex,
    max_denom: &Integer,
    tol_bits: u32,
) -> Option<Rational> {
    // Fast path: integral value.
    if let Some(n) = complex_round_to_integer_if_close(z, tol_bits) {
        return Some(Rational::from_integer(n));
    }
    // The imaginary part must vanish for a rational value.
    if !near_zero_bf(&z.im, tol_bits) {
        return None;
    }
    rational_reconstruction(&z.re, max_denom, tol_bits)
}

/// All rational coset values of the resolvent (diagnostic: lets callers report
/// how many resolvent roots are rational, the orbit structure, etc.).
pub fn rational_values(
    res: &RelativeResolvent,
    max_denom: &Integer,
    tol_bits: u32,
) -> Vec<(Perm, Rational)> {
    res.coset_values
        .iter()
        .filter_map(|cv| {
            value_to_rational(&cv.value, max_denom, tol_bits).map(|q| (cv.rep.clone(), q))
        })
        .collect()
}
