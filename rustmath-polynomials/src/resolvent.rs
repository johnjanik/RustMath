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

// --------------------------------------------------------------------------- //
// General k-subset resolvent via power sums (exact, any k)
// --------------------------------------------------------------------------- //

/// Truncated product of two power series over `ℚ` (coefficients to degree `n`).
fn series_mul(a: &[Rational], b: &[Rational], n: usize) -> Vec<Rational> {
    let mut c = vec![rzero(); n + 1];
    for (i, ai) in a.iter().enumerate().take(n + 1) {
        if *ai == rzero() {
            continue;
        }
        for (j, bj) in b.iter().enumerate().take(n + 1 - i) {
            c[i + j] = c[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    c
}

/// Power sums `π_r = Σ αᵢ^r` (`r = 0..=upto`) of the roots of monic `f`
/// (little-endian, `ℚ`), via Newton's identities. `eⱼ = (−1)ʲ a_{n−j}`.
fn power_sums(f: &[Rational], upto: usize) -> Vec<Rational> {
    let n = f.len() - 1;
    let e = |j: usize| -> Rational {
        let a = f[n - j].clone();
        if j % 2 == 0 {
            a
        } else {
            -a
        }
    };
    let mut p = vec![rzero(); upto + 1];
    p[0] = rq(n as i64);
    for r in 1..=upto {
        let mut acc = rzero();
        let sum_lim = (r - 1).min(n);
        for j in 1..=sum_lim {
            let term = e(j) * p[r - j].clone();
            acc = if j % 2 == 1 { acc + term } else { acc - term };
        }
        if r <= n {
            let term = e(r) * rq(r as i64);
            acc = if r % 2 == 1 { acc + term } else { acc - term };
        }
        p[r] = acc;
    }
    p
}

/// The exact **k-subset-sum resolvent** `∏_{|S|=k}(Y − Σ_{i∈S} αᵢ)` of a monic
/// `f ∈ ℤ[x]` of degree `n`, of degree `C(n, k)`, returned little-endian over `ℤ`.
///
/// This is the absolute resolvent for descending `Sₙ` to the stabilizer of a
/// `k`-set (the intransitive maximal subgroup `Sₖ × S_{n−k}`) — a genuine Stauduhar
/// resolvent. Its irreducible-factor degrees are the orbit lengths of `Gal(f)` on
/// `k`-subsets of the roots (when the subset-sums are distinct), matched on the
/// group side by [`rustmath_groups`]`::ksubset_orbits::orbit_lengths_on_ksubsets`.
///
/// Built exactly from the power sums of the subset-sums:
/// `Σ_{|S|=k} e^{t·Σ_{i∈S}αᵢ} = e_k(e^{tα₁},…,e^{tαₙ})`, computing `e_k` of the
/// `yᵢ = e^{tαᵢ}` via Newton's identities on `P_j = Σᵢ yᵢ^j = Σ_r (jʳ π_r / r!) tʳ`,
/// then recovering the resolvent coefficients from its power sums `q_m = m!·[tᵐ]e_k`.
///
/// `k` must satisfy `1 ≤ k ≤ n`; panics if `C(n,k) > 4096` (resolvent too large to
/// build exactly) or if `f` is not monic.
pub fn subset_sum_resolvent(f: &[Integer], k: usize) -> Vec<Integer> {
    let n = f.len() - 1;
    assert!(k >= 1 && k <= n, "need 1 ≤ k ≤ n");
    assert!(f[n] == Integer::one(), "subset_sum_resolvent requires a monic f");
    let big_n = binom_usize(n, k);
    assert!(big_n <= 4096, "C(n,k) = {big_n} too large to build exactly");
    let fq = to_q(f);

    // Factorials 0!..N! as ℚ, and π_0..π_N.
    let nn = big_n;
    let mut fact = vec![rq(1); nn + 1];
    for r in 1..=nn {
        fact[r] = fact[r - 1].clone() * rq(r as i64);
    }
    let pi = power_sums(&fq, nn);

    // P_j = Σ_r (j^r π_r / r!) t^r, for j = 1..=k.
    let pseries: Vec<Vec<Rational>> = (1..=k)
        .map(|j| {
            (0..=nn)
                .map(|r| {
                    let jp = Integer::from(j as i64).pow(r as u32);
                    pi[r].clone() * Rational::new(jp, fact_int(r)).unwrap()
                })
                .collect()
        })
        .collect();

    // e_i (series) via Newton: i·e_i = Σ_{j=1}^{i} (−1)^{j−1} e_{i−j} P_j.
    let mut es: Vec<Vec<Rational>> = Vec::with_capacity(k + 1);
    let mut e0 = vec![rzero(); nn + 1];
    e0[0] = rq(1);
    es.push(e0);
    for i in 1..=k {
        let mut acc = vec![rzero(); nn + 1];
        for j in 1..=i {
            let prod = series_mul(&es[i - j], &pseries[j - 1], nn);
            for r in 0..=nn {
                acc[r] = if j % 2 == 1 {
                    acc[r].clone() + prod[r].clone()
                } else {
                    acc[r].clone() - prod[r].clone()
                };
            }
        }
        let inv_i = Rational::new(Integer::one(), Integer::from(i as i64)).unwrap();
        for r in 0..=nn {
            acc[r] = acc[r].clone() * inv_i.clone();
        }
        es.push(acc);
    }

    // q_m = m! · [t^m] e_k = power sums of the subset-sums.
    let ek = &es[k];
    let q: Vec<Rational> = (0..=nn).map(|m| fact[m].clone() * ek[m].clone()).collect();

    // Recover elementary symmetric E_m of the subset-sums from q via Newton.
    let mut bige = vec![rzero(); nn + 1];
    bige[0] = rq(1);
    for m in 1..=nn {
        let mut acc = rzero();
        for j in 1..=m {
            let term = bige[m - j].clone() * q[j].clone();
            acc = if j % 2 == 1 { acc + term } else { acc - term };
        }
        bige[m] = acc * Rational::new(Integer::one(), Integer::from(m as i64)).unwrap();
    }

    // R(Y) = ∏(Y − s_S) = Σ_m (−1)^m E_m Y^{N−m}; little-endian coeff[N−m]=(−1)^m E_m.
    let mut coeffs = vec![rzero(); nn + 1];
    for m in 0..=nn {
        coeffs[nn - m] = if m % 2 == 0 {
            bige[m].clone()
        } else {
            -bige[m].clone()
        };
    }
    trim_z(to_z(&coeffs))
}

fn fact_int(r: usize) -> Integer {
    let mut f = Integer::one();
    for i in 1..=r {
        f = f * Integer::from(i as i64);
    }
    f
}

fn binom_usize(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num = 1u128;
    let mut den = 1u128;
    for i in 0..k {
        num *= (n - i) as u128;
        den *= (i + 1) as u128;
    }
    (num / den) as usize
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
    fn subset_sum_k2_agrees_with_resultant_method() {
        // Two independent algorithms (power-sum vs resultant) must agree for k=2.
        for f in [
            iz(&[-2, 0, 0, 1]),       // x³−2
            iz(&[1, 1, 0, 0, 1]),     // x⁴+x+1
            iz(&[-2, 0, 0, 0, 1]),    // x⁴−2
            iz(&[-1, -1, 0, 0, 0, 1]),// x⁵−x−1
            iz(&[7, 0, -3, 0, 1]),    // x⁴−3x²+7
        ] {
            assert_eq!(subset_sum_resolvent(&f, 2), pair_sum_resolvent(&f), "k=2 mismatch for {f:?}");
        }
    }

    #[test]
    fn subset_sum_k3_matches_pari_gp() {
        // x⁴+x+1, k=3: 3-subset sums = e₁−αᵢ = −αᵢ ⇒ Y⁴ − Y + 1.
        assert_eq!(subset_sum_resolvent(&iz(&[1, 1, 0, 0, 1]), 3), iz(&[1, -1, 0, 0, 1]));
        // x⁵−x−1, k=3 (degree C(5,3)=10), from PARI/GP.
        assert_eq!(
            subset_sum_resolvent(&iz(&[-1, -1, 0, 0, 0, 1]), 3),
            iz(&[-1, -4, -4, 0, 0, -11, 3, 0, 0, 0, 1])
        );
        // x⁵−x−1, k=2 from PARI/GP (also cross-checks k=2 path).
        assert_eq!(
            subset_sum_resolvent(&iz(&[-1, -1, 0, 0, 0, 1]), 2),
            iz(&[-1, 4, -4, 0, 0, 11, 3, 0, 0, 0, 1])
        );
    }

    #[test]
    fn subset_sum_k1_is_f_and_complement_reflects() {
        // k=1: the 1-subset sums are the roots ⇒ resolvent = f.
        let f = iz(&[-1, -1, 0, 0, 0, 1]); // x⁵−x−1
        assert_eq!(subset_sum_resolvent(&f, 1), f);
        // k=n: the single n-subset sum is the trace = −a_{n−1}; degree-1 (Y − tr).
        let r = subset_sum_resolvent(&iz(&[1, 1, 0, 0, 1]), 4); // trace 0
        assert_eq!(r, iz(&[0, 1]));
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
