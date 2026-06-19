//! Resolvent polynomials for Galois-group determination (Module 18, Lagrange /
//! Stauduhar family).
//!
//! Let `f ∈ ℤ[x]` be monic, separable, of degree `n`, with roots `α₁,…,αₙ`. For a
//! family `𝒮` of subsets of `{1,…,n}` permuted by `Sₙ`, the **linear (Lagrange)
//! resolvent** is
//! ```text
//!     R(Y) = ∏_{S ∈ 𝒮} (Y − Σ_{i∈S} αᵢ).
//! ```
//! Its coefficients are symmetric in the `αᵢ`, hence (for monic integral `f`)
//! integers. The degrees of the irreducible factors of `R` over `ℚ` are exactly the
//! orbit lengths of `Gal(f) ⊆ Sₙ` acting on `𝒮` — the central invariant of the
//! Stauduhar method. Two transitive groups with identical cycle-type profiles (a
//! Frobenius-blind class) are routinely separated by such an orbit signature.
//!
//! We build resolvents **exactly**, never numerically: at degree 24 the resolvent
//! coefficients dwarf `f64`. The canonical case — the family of 2-element subsets —
//! uses the resultant identity
//! ```text
//!     Res_x(f(x), f(Y−x)) = ∏_{i,j} (Y − αᵢ − αⱼ)
//!                         = ∏_i (Y − 2αᵢ) · [∏_{i<j} (Y − αᵢ − αⱼ)]²,
//! ```
//! so the **pair-sum resolvent** `∏_{i<j}(Y − αᵢ − αⱼ)` is the exact polynomial
//! square root of `Res_x(f(x), f(Y−x)) / ∏_i(Y − 2αᵢ)`, with
//! `∏_i(Y − 2αᵢ) = 2ⁿ f(Y/2)`. Both pieces are built with the validated bivariate
//! resultant ([`crate::bivariate`]) and exact `ℚ[Y]` square root.
//!
//! The degree-2 **discriminant resolvent** `Y² − disc(f)` decides `Gal(f) ⊆ Aₙ`: it
//! splits over `ℚ` (i.e. `disc` is a square) iff every element of the Galois group is
//! an even permutation. Built on [`crate::disc::discriminant`].
//!
//! Coefficient convention throughout: little-endian `Vec`, `c[i]` = coeff of `xⁱ`.

use crate::bivariate;
use crate::disc::discriminant;
use crate::univariate::UnivariatePolynomial;
use rustmath_core::Result;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;

fn rq(n: i64) -> Rational {
    Rational::from_i64(n)
}

fn rzero() -> Rational {
    rq(0)
}

fn deg_q(p: &[Rational]) -> i64 {
    let mut n = p.len();
    while n > 0 && p[n - 1] == rzero() {
        n -= 1;
    }
    n as i64 - 1
}

/// Binomial coefficient `C(k, a)` as a `Rational` (`k ≤ degree`, fits easily).
fn binom(k: usize, a: usize) -> Rational {
    if a > k {
        return rzero();
    }
    let mut num = Integer::one();
    let mut den = Integer::one();
    for i in 0..a {
        num = num * Integer::from((k - i) as i64);
        den = den * Integer::from((i + 1) as i64);
    }
    Rational::new(num, den).expect("nonzero denominator")
}

/// Exact quotient `a / b` over `ℚ[Y]`; panics if the division is not exact.
fn exact_div_q(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    let db = deg_q(b);
    assert!(db >= 0, "division by zero polynomial");
    let lcb_inv = b[db as usize].reciprocal().expect("nonzero leading coeff");
    let mut r = a.to_vec();
    let mut quo = vec![rzero(); (deg_q(a) - db + 1).max(0) as usize];
    while deg_q(&r) >= db {
        let dr = deg_q(&r) as usize;
        let coeff = r[dr].clone() * lcb_inv.clone();
        let shift = dr - db as usize;
        quo[shift] = coeff.clone();
        for j in 0..b.len() {
            r[j + shift] = r[j + shift].clone() - coeff.clone() * b[j].clone();
        }
        while r.len() > 1 && *r.last().unwrap() == rzero() {
            r.pop();
        }
    }
    assert!(deg_q(&r) < 0, "non-exact polynomial division in resolvent");
    quo
}

/// Convert an integer little-endian polynomial to `ℚ` coefficients.
fn to_q(f: &[Integer]) -> Vec<Rational> {
    f.iter().map(|c| Rational::from_integer(c.clone())).collect()
}

/// Convert a rational little-endian polynomial to a **primitive integer** one,
/// requiring every coefficient to be an integer (panics otherwise).
fn to_z(f: &[Rational]) -> Vec<Integer> {
    f.iter()
        .map(|c| {
            assert!(c.is_integer(), "resolvent coefficient is not integral");
            c.numerator().clone()
        })
        .collect()
}

fn trim_z(mut f: Vec<Integer>) -> Vec<Integer> {
    while f.len() > 1 && f.last().map(|c| c.is_zero()).unwrap_or(false) {
        f.pop();
    }
    f
}

/// The **discriminant resolvent** `Y² − disc(f)` (little-endian `[−disc, 0, 1]`).
/// `Gal(f) ⊆ Aₙ` iff this splits over `ℚ`, i.e. iff `disc(f)` is a perfect square.
pub fn discriminant_resolvent(f: &[Integer]) -> Vec<Integer> {
    let d = discriminant(f);
    vec![-d, Integer::zero(), Integer::one()]
}

/// Decide whether `Gal(f) ⊆ Aₙ` (every automorphism is an even permutation), i.e.
/// whether `disc(f)` is a nonzero perfect square. Requires `f` separable.
pub fn galois_in_alternating(f: &[Integer]) -> bool {
    let d = discriminant(f);
    !d.is_zero() && d.is_perfect_square()
}

/// The exact **pair-sum resolvent** `∏_{i<j}(Y − αᵢ − αⱼ)` of a monic separable
/// `f ∈ ℤ[x]`, of degree `C(n, 2)`, returned little-endian over `ℤ`.
///
/// Galois meaning: the irreducible-factor degrees of this resolvent are the orbit
/// lengths of `Gal(f)` on unordered pairs of roots. A single factor of degree
/// `C(n,2)` ⟺ 2-homogeneous action; the partition refines as the group shrinks —
/// the discriminator for many Frobenius-blind degree-`n` classes.
pub fn pair_sum_resolvent(f: &[Integer]) -> Vec<Integer> {
    let n = f.len() - 1; // degree
    assert!(n >= 2, "pair-sum resolvent needs degree ≥ 2");
    assert!(
        f[n] == Integer::one(),
        "pair-sum resolvent requires a monic f (got non-monic leading coeff)"
    );

    // f(x) as a bivariate (constant in t): fbiv[i] = [f_i].
    let fbiv: Vec<Vec<Rational>> =
        f.iter().map(|c| vec![Rational::from_integer(c.clone())]).collect();

    // g(x) = f(t − x): coeff of x^{k−a} t^a in the term f_k is f_k·C(k,a)·(−1)^{k−a}.
    let mut gbiv: Vec<Vec<Rational>> = vec![vec![rzero(); n + 1]; n + 1];
    for k in 0..=n {
        if f[k].is_zero() {
            continue;
        }
        let fk = Rational::from_integer(f[k].clone());
        for a in 0..=k {
            let xpow = k - a;
            let tpow = a;
            let mut term = fk.clone() * binom(k, a);
            if (k - a) % 2 == 1 {
                term = -term;
            }
            gbiv[xpow][tpow] = gbiv[xpow][tpow].clone() + term;
        }
    }

    // Res_x(f(x), f(Y−x)) ∈ ℚ[Y], degree n².
    let res = bivariate::resultant_in_t(&fbiv, &gbiv);

    // ∏_i(Y − 2αᵢ) = 2ⁿ f(Y/2):  coeff of Y^k is f_k · 2^{n−k}.
    let two = Integer::from(2);
    let dpoly: Vec<Rational> = (0..=n)
        .map(|k| {
            let scale = two.pow((n - k) as u32);
            Rational::from_integer(f[k].clone() * scale)
        })
        .collect();

    // [pair-sum resolvent]² = res / dpoly; take the exact ℚ[Y] square root.
    let squared = exact_div_q(&res, &dpoly);
    let root = bivariate::poly_sqrt(&squared)
        .expect("pair-sum resolvent: quotient is not a perfect square (f not separable?)");
    trim_z(to_z(&root))
}

/// Orbit signature of a resolvent: the sorted multiset of irreducible-factor
/// degrees of `resolvent` over `ℚ` (constant factors dropped, multiplicities kept).
/// For a separable resolvent these are the Galois orbit lengths on the underlying
/// subset family.
pub fn resolvent_orbit_signature(resolvent: &[Integer]) -> Result<Vec<usize>> {
    let poly = UnivariatePolynomial::new(resolvent.to_vec());
    let factors = crate::factorization::factor_over_integers(&poly)?;
    let mut degs: Vec<usize> = Vec::new();
    for (g, mult) in factors {
        if let Some(d) = g.degree() {
            if d >= 1 {
                for _ in 0..mult {
                    degs.push(d);
                }
            }
        }
    }
    degs.sort_unstable();
    Ok(degs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iz(v: &[i64]) -> Vec<Integer> {
        v.iter().map(|&x| Integer::from(x)).collect()
    }

    #[test]
    fn discriminant_resolvent_detects_alternating() {
        // x³ − 3x − 1: disc = 81 = 9², Galois group C₃ ⊆ A₃.
        assert!(galois_in_alternating(&iz(&[-1, -3, 0, 1])));
        assert_eq!(discriminant_resolvent(&iz(&[-1, -3, 0, 1])), iz(&[-81, 0, 1]));
        // x³ − 2: disc = −108, not a square, Galois group S₃ ⊄ A₃.
        assert!(!galois_in_alternating(&iz(&[-2, 0, 0, 1])));
    }

    #[test]
    fn pair_sum_resolvent_cubic_x3_minus_2() {
        // Roots sum to 0, so αᵢ+αⱼ = −αₖ; the 3 pair-sums are {−α₁,−α₂,−α₃},
        // giving ∏(Y+αₖ) = −f(−Y) = Y³ + 2.
        let r = pair_sum_resolvent(&iz(&[-2, 0, 0, 1]));
        assert_eq!(r, iz(&[2, 0, 0, 1]));
    }

    #[test]
    fn pair_sum_resolvent_degree_is_binom_n_2() {
        // x⁴ + x + 1: C(4,2) = 6 pair-sums ⇒ degree-6 resolvent.
        let r = pair_sum_resolvent(&iz(&[1, 1, 0, 0, 1]));
        assert_eq!(r.len() - 1, 6);
        // leading coeff monic
        assert_eq!(*r.last().unwrap(), Integer::one());
    }

    #[test]
    fn pair_sum_resolvent_matches_pari_gp() {
        // Cross-checked against PARI/GP polresultant(f, f(Y-x))/(2^n f(Y/2)), sqrt.
        // x⁴ + x + 1 → Y⁶ − 4Y² − 1
        assert_eq!(
            pair_sum_resolvent(&iz(&[1, 1, 0, 0, 1])),
            iz(&[-1, 0, -4, 0, 0, 0, 1])
        );
        // x⁴ − 2 → Y⁶ + 8Y²
        assert_eq!(
            pair_sum_resolvent(&iz(&[-2, 0, 0, 0, 1])),
            iz(&[0, 0, 8, 0, 0, 0, 1])
        );
        // x⁵ − x − 1 → degree-10 resolvent
        assert_eq!(
            pair_sum_resolvent(&iz(&[-1, -1, 0, 0, 0, 1])),
            iz(&[-1, 4, -4, 0, 0, 11, 3, 0, 0, 0, 1])
        );
    }

    #[test]
    fn pair_sum_resolvent_reducible_for_imprimitive() {
        // x⁴ − 2 has Galois group D₄ (order 8), imprimitive with blocks {α,−α}.
        // Its pair-sum resolvent (degree 6) factors non-trivially over ℚ.
        let r = pair_sum_resolvent(&iz(&[-2, 0, 0, 0, 1]));
        let sig = resolvent_orbit_signature(&r).unwrap();
        assert_eq!(sig.iter().sum::<usize>(), 6);
        assert!(sig.len() > 1, "D₄ acts intransitively on pairs: {:?}", sig);
    }
}
