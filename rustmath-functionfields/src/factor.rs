//! Factorization of polynomials over the rational function field `ℚ(t)`.
//!
//! # Method (Trager-style: norm + factor over ℚ + recombination)
//!
//! To factor `F ∈ ℚ(t)[x]` we exploit that `ℚ(t)` is the field of fractions of
//! `ℚ[t]`, so factoring `F` is equivalent to factoring the cleared, `x`-primitive
//! bivariate polynomial `F̃ ∈ ℚ[t][x]` over `ℚ` (Gauss's lemma) and discarding
//! pure-`t` content. We do this without a standalone bivariate factorizer by the
//! classical *evaluate–factor–recombine* scheme:
//!
//! 1. **Squarefree.** Reduce to the squarefree part (Trager assumes a separable
//!    input; multiplicities are handled by the squarefree decomposition outside).
//! 2. **Norm via specialization.** Choose a *good* place `t = a ∈ ℚ`: the leading
//!    `x`-coefficient does not vanish, no coefficient has a pole, and `F(a, x)` is
//!    separable over `ℚ`. The univariate `F(a, x) ∈ ℚ[x]` is the "norm".
//! 3. **Factor over ℚ.** Factor `F(a, x)` with the integer Zassenhaus factorizer
//!    reused from `rustmath-polynomials`.
//! 4. **Recombine by trial division in `ℚ(t)[x]`.** Each `ℚ(t)`-irreducible factor
//!    of `F` reduces, at the good place `a`, to a product of some of the
//!    `ℚ`-factors. We search subsets of the local factors, **lift** the product to
//!    a candidate `G ∈ ℚ(t)[x]` by multi-point interpolation in `t`, and accept it
//!    iff `G | F` exactly in `ℚ(t)[x]` (division there is exact — `ℚ(t)` is a
//!    field). Trial division makes the recombination *certain*; the lift only has
//!    to propose candidates.
//!
//! The recombination is worst-case exponential in the number of local factors
//! (as is true of all evaluate–recombine factorizers without a Hensel/LLL bound);
//! this is acceptable for the modest degrees of regular-cover scanning. See
//! `lib.rs` for what is implemented vs deferred.

use crate::function_field::FfPoly;
use crate::ratfunc::{QtPoly, RationalFunction};
use rustmath_core::Ring;
use rustmath_polynomials::factorization::factor_over_integers;
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

type QxPoly = UnivariatePolynomial<Rational>;

/// Clear denominators of a `ℚ[x]` polynomial to obtain an integer polynomial of
/// the same roots, scaling by the lcm of denominators. Returns the integer
/// polynomial (little-endian).
fn clear_denominators_q(p: &QxPoly) -> UnivariatePolynomial<rustmath_integers::Integer> {
    use rustmath_integers::Integer;
    // lcm of denominators
    let mut l = Integer::one();
    for c in p.coefficients() {
        let d = c.denominator().clone();
        let g = l.gcd(&d);
        l = l.clone() / g * d;
    }
    let coeffs: Vec<Integer> = p
        .coefficients()
        .iter()
        .map(|c| {
            let scaled = c.clone() * Rational::from_integer(l.clone());
            // scaled is now an integer
            scaled.numerator().clone()
        })
        .collect();
    UnivariatePolynomial::new(coeffs)
}

/// Factor a `ℚ[x]` polynomial into monic irreducible factors over `ℚ`
/// (multiplicities flattened: a factor of multiplicity m appears m times),
/// ignoring the constant content. Each returned factor is monic.
fn factor_over_q(p: &QxPoly) -> Vec<QxPoly> {
    let int_poly = clear_denominators_q(p);
    let factors = match factor_over_integers(&int_poly) {
        Ok(f) => f,
        Err(_) => return vec![p.clone().make_monic()],
    };
    let mut out = Vec::new();
    for (g, mult) in factors {
        // Skip constant factors (the content).
        if g.degree().unwrap_or(0) == 0 {
            continue;
        }
        // Re-express over ℚ and make monic.
        let gq: QxPoly = UnivariatePolynomial::new(
            g.coefficients()
                .iter()
                .map(|c| Rational::from_integer(c.clone()))
                .collect(),
        )
        .make_monic();
        for _ in 0..mult {
            out.push(gq.clone());
        }
    }
    if out.is_empty() {
        out.push(p.clone().make_monic());
    }
    out
}

/// Monic gcd in `ℚ(t)[x]` (Euclidean: `ℚ(t)` is a field).
fn ff_gcd(a: &FfPoly, b: &FfPoly) -> FfPoly {
    let mut a = a.clone();
    let mut b = b.clone();
    while !b.is_zero() {
        let (_, r) = a.div_rem(&b).unwrap();
        a = b;
        b = r;
    }
    if a.is_zero() {
        a
    } else {
        a.make_monic()
    }
}

/// Evaluate the `x`-coefficients of `F ∈ ℚ(t)[x]` at `t = a`, yielding `F(a, x) ∈ ℚ[x]`.
/// Returns `None` on a coefficient pole.
fn specialize_to_qx(f: &FfPoly, a: &Rational) -> Option<QxPoly> {
    let deg = f.degree()?;
    let mut coeffs = Vec::with_capacity(deg + 1);
    for i in 0..=deg {
        coeffs.push(f.coeff(i).evaluate(a)?);
    }
    Some(UnivariatePolynomial::new(coeffs))
}

/// Multiply a list of monic `ℚ[x]` factors together.
fn product_qx(factors: &[&QxPoly]) -> QxPoly {
    let mut p = UnivariatePolynomial::new(vec![Rational::one()]);
    for f in factors {
        p = p * (*f).clone();
    }
    p
}

/// Lift a target `x`-degree-`d` monic factor, known modulo each of several good
/// places, to a candidate `G ∈ ℚ(t)[x]` by Lagrange interpolation of each
/// `x`-coefficient as a function of `t`.
///
/// `samples` is a list of `(a_j, g_j)` where `g_j ∈ ℚ[x]` is the proposed monic
/// factor at place `t = a_j`. All `g_j` must share the same `x`-degree `d`.
/// Returns the interpolated `G` (monic in `x`), or `None` if degrees disagree.
fn interpolate_factor(samples: &[(Rational, QxPoly)]) -> Option<FfPoly> {
    let d = samples[0].1.degree()?;
    for (_, g) in samples {
        if g.degree() != Some(d) {
            return None;
        }
    }
    // For each x-power i = 0..d, interpolate the ℚ-values { g_j coeff(i) } over
    // the nodes { a_j } into a polynomial in t, then embed as a ℚ(t) constant.
    let nodes: Vec<Rational> = samples.iter().map(|(a, _)| a.clone()).collect();
    let mut ff_coeffs: Vec<RationalFunction> = Vec::with_capacity(d + 1);
    for i in 0..=d {
        let values: Vec<Rational> =
            samples.iter().map(|(_, g)| g.coeff(i).clone()).collect();
        let poly_t = lagrange_interpolate(&nodes, &values)?;
        ff_coeffs.push(RationalFunction::from_poly(poly_t));
    }
    Some(UnivariatePolynomial::new(ff_coeffs))
}

/// Lagrange interpolation: given distinct nodes `x_j` and values `y_j`, return the
/// unique polynomial `p ∈ ℚ[t]` of degree `< n` with `p(x_j) = y_j`.
fn lagrange_interpolate(nodes: &[Rational], values: &[Rational]) -> Option<QtPoly> {
    let n = nodes.len();
    if n != values.len() || n == 0 {
        return None;
    }
    let mut result = QtPoly::zero();
    for j in 0..n {
        // Basis polynomial L_j(t) = prod_{k != j} (t - x_k)/(x_j - x_k)
        let mut basis = QtPoly::one();
        let mut denom = Rational::one();
        for k in 0..n {
            if k == j {
                continue;
            }
            // (t - x_k)
            let factor = UnivariatePolynomial::new(vec![-nodes[k].clone(), Rational::one()]);
            basis = basis * factor;
            let diff = nodes[j].clone() - nodes[k].clone();
            if diff.is_zero() {
                return None; // repeated node
            }
            denom = denom * diff;
        }
        let scale = values[j].clone() / denom;
        result = result + basis.scalar_mul(&scale);
    }
    Some(result)
}

/// Factor a monic, squarefree `F ∈ ℚ(t)[x]` into its monic irreducible factors
/// over `ℚ(t)`. Internal worker used by [`factor_over_qt`].
fn factor_squarefree(f: &FfPoly) -> Vec<FfPoly> {
    let deg = match f.degree() {
        Some(d) if d > 0 => d,
        _ => return vec![f.clone()],
    };
    if deg == 1 {
        return vec![f.clone()];
    }

    // Collect several good places so interpolation can recover t-dependence up to
    // the degree present in the coefficients.
    let mut places: Vec<(Rational, QxPoly)> = Vec::new();
    let mut local_factorizations: Vec<Vec<QxPoly>> = Vec::new();
    let mut tried = 0i64;
    let mut n = 0i64;
    // Bound the number of interpolation nodes by the max coefficient t-degree + a
    // margin; gather that many good, *factor-shape-consistent* places.
    let max_t_deg = max_coeff_t_degree(f);
    let needed_nodes = max_t_deg + 2;

    while places.len() < needed_nodes && tried < 200 {
        tried += 1;
        let a = if n == 0 {
            Rational::from_i64(0)
        } else if n % 2 == 1 {
            Rational::from_i64((n + 1) / 2)
        } else {
            Rational::from_i64(-(n / 2))
        };
        n += 1;
        if let Some(fa) = specialize_to_qx(f, &a) {
            if fa.degree() == Some(deg) && fa.is_square_free() {
                let local = factor_over_q(&fa);
                places.push((a, fa));
                local_factorizations.push(local);
            }
        }
    }

    if places.is_empty() {
        return vec![f.clone()];
    }

    // If at the first good place the polynomial is irreducible over ℚ, then F is
    // irreducible over ℚ(t) (a ℚ(t)-factorization would specialize to a proper
    // ℚ-factorization at every good place).
    if local_factorizations[0].len() == 1 {
        return vec![f.clone()];
    }

    // Recombination at the *first* good place. Subsets of its local factors whose
    // interpolated lift divides F are the true ℚ(t)-factors.
    let local0 = &local_factorizations[0];
    let m = local0.len();

    let mut remaining = f.clone();
    let mut found: Vec<FfPoly> = Vec::new();
    let mut used = vec![false; m];

    // Try subsets by increasing size so we peel off the lowest-degree irreducible
    // factors first (each true factor is a minimal divisible subset).
    for size in 1..m {
        let mut combo = vec![0usize; size];
        let mut go = true;
        for (i, c) in combo.iter_mut().enumerate() {
            *c = i;
        }
        while go {
            if combo.iter().all(|&i| !used[i]) {
                // Generate every candidate lift consistent with this reference
                // subset and accept the first that *exactly* divides F. Trial
                // division is the certificate, so a wrong matching is harmless.
                let candidates = build_candidates(&combo, &places, &local_factorizations);
                for cand in candidates {
                    if cand.degree().unwrap_or(0) == 0 {
                        continue;
                    }
                    if divides(&cand, &remaining) {
                        let (q, _) = remaining.div_rem(&cand).unwrap();
                        remaining = q.make_monic();
                        for &i in &combo {
                            used[i] = true;
                        }
                        found.push(cand.make_monic());
                        break;
                    }
                }
            }
            go = next_combination(&mut combo, m);
        }
        if remaining.degree().unwrap_or(0) == 0 {
            break;
        }
    }

    // Whatever remains (the unused factors) is itself irreducible (or the whole
    // thing if no proper factor was found).
    if remaining.degree().unwrap_or(0) > 0 {
        found.push(remaining.make_monic());
    }
    if found.is_empty() {
        found.push(f.clone());
    }
    found
}

/// Generate candidate `ℚ(t)[x]` factors for a subset `combo` of the reference
/// place's local factors.
///
/// At the reference place the candidate image is the product of `local0[combo]`,
/// of some `x`-degree `d`. At every *other* collected place we don't know which
/// local factors correspond, so we enumerate **all** degree-`d` subset products
/// there, take the Cartesian product of those choices across places (bounded),
/// and interpolate each combination of per-place images into a `ℚ(t)[x]`
/// candidate. Exactly one combination interpolates to the true factor; the caller
/// confirms it by exact division, so spurious candidates are simply discarded.
fn build_candidates(
    combo: &[usize],
    places: &[(Rational, QxPoly)],
    local_factorizations: &[Vec<QxPoly>],
) -> Vec<FfPoly> {
    let refs: Vec<&QxPoly> = combo.iter().map(|&i| &local_factorizations[0][i]).collect();
    let ref_img = product_qx(&refs);
    let d = match ref_img.degree() {
        Some(d) => d,
        None => return Vec::new(),
    };

    // Per-place lists of degree-d subset products. Place 0 is fixed to ref_img.
    let mut per_place_images: Vec<Vec<QxPoly>> = Vec::with_capacity(places.len());
    per_place_images.push(vec![ref_img]);
    for j in 1..places.len() {
        let imgs = degree_d_subset_products(&local_factorizations[j], d);
        if imgs.is_empty() {
            // This place offers no consistent image; drop it from interpolation.
            continue;
        }
        per_place_images.push(imgs);
    }

    // Cartesian product of one image per place, bounded to avoid blow-up.
    let nodes: Vec<Rational> = (0..per_place_images.len())
        .map(|j| places[j].0.clone())
        .collect();

    let mut candidates = Vec::new();
    let total: usize = per_place_images.iter().map(|v| v.len()).product();
    if total == 0 || total > 4096 {
        // Fall back to single-node interpolation (correct when the true factor has
        // constant-in-t coefficients).
        if let Some(g) = interpolate_factor(&[(nodes[0].clone(), per_place_images[0][0].clone())]) {
            candidates.push(g);
        }
        return candidates;
    }

    let mut idx = vec![0usize; per_place_images.len()];
    loop {
        let samples: Vec<(Rational, QxPoly)> = (0..per_place_images.len())
            .map(|j| (nodes[j].clone(), per_place_images[j][idx[j]].clone()))
            .collect();
        if let Some(g) = interpolate_factor(&samples) {
            candidates.push(g);
        }
        // Advance the mixed-radix counter.
        let mut k = per_place_images.len();
        loop {
            if k == 0 {
                return candidates;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < per_place_images[k].len() {
                break;
            }
            idx[k] = 0;
        }
    }
}

/// All distinct products of subsets of `factors` whose total `x`-degree is `d`.
fn degree_d_subset_products(factors: &[QxPoly], d: usize) -> Vec<QxPoly> {
    let degs: Vec<usize> = factors.iter().map(|f| f.degree().unwrap_or(0)).collect();
    let m = factors.len();
    let mut out: Vec<QxPoly> = Vec::new();
    if m > 24 {
        return out; // guard against pathological masks
    }
    for mask in 1u32..(1u32 << m) {
        let total: usize = (0..m)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| degs[i])
            .sum();
        if total == d {
            let sel: Vec<&QxPoly> = (0..m)
                .filter(|&i| mask & (1 << i) != 0)
                .map(|i| &factors[i])
                .collect();
            out.push(product_qx(&sel));
        }
    }
    out
}

/// Exact divisibility test in `ℚ(t)[x]`.
fn divides(g: &FfPoly, f: &FfPoly) -> bool {
    if g.is_zero() {
        return false;
    }
    let (_, r) = f.div_rem(g).unwrap();
    r.is_zero()
}

/// Maximum `t`-degree appearing across the numerators/denominators of `F`'s
/// `x`-coefficients — an upper bound for the interpolation node count.
fn max_coeff_t_degree(f: &FfPoly) -> usize {
    let mut m = 0usize;
    for c in f.coefficients() {
        let dn = c.numerator().degree().unwrap_or(0);
        let dd = c.denominator().degree().unwrap_or(0);
        m = m.max(dn).max(dd);
    }
    m
}

/// Advance `combo` (a strictly increasing length-`k` subset of `0..n`) to the next
/// combination in lexicographic order. Returns `false` when exhausted.
fn next_combination(combo: &mut [usize], n: usize) -> bool {
    let k = combo.len();
    if k == 0 {
        return false;
    }
    let mut i = k;
    while i > 0 {
        i -= 1;
        if combo[i] != i + n - k {
            combo[i] += 1;
            for j in (i + 1)..k {
                combo[j] = combo[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// Factor `F ∈ ℚ(t)[x]` into monic irreducible factors over `ℚ(t)`, with
/// multiplicities. Returns `(content, factors)` where `content ∈ ℚ(t)` is the
/// leading coefficient pulled out so every returned factor is monic, and
/// `factors` is a list of `(irreducible, multiplicity)` pairs.
///
/// The product of `content · ∏ factor_i^{mult_i}` equals the input `F`.
pub fn factor_over_qt(f: &FfPoly) -> (RationalFunction, Vec<(FfPoly, usize)>) {
    if f.is_zero() {
        return (RationalFunction::zero(), Vec::new());
    }
    let deg = f.degree().unwrap();
    let content = f.leading_coefficient().cloned().unwrap();
    if deg == 0 {
        return (content, Vec::new());
    }
    let monic = f.clone().make_monic();

    // Squarefree decomposition over ℚ(t): f = ∏ g_k^k with g_k squarefree.
    let sqfree_factors = squarefree_decomposition(&monic);

    let mut result: Vec<(FfPoly, usize)> = Vec::new();
    for (g, mult) in sqfree_factors {
        for irr in factor_squarefree(&g) {
            // Merge equal irreducibles (raise multiplicity).
            if let Some(entry) = result.iter_mut().find(|(p, _)| *p == irr) {
                entry.1 += mult;
            } else {
                result.push((irr, mult));
            }
        }
    }
    (content, result)
}

/// Squarefree decomposition of monic `F ∈ ℚ(t)[x]`: returns `(g_k, k)` with
/// `F = ∏ g_k^k`, each `g_k` squarefree and pairwise coprime.
fn squarefree_decomposition(f: &FfPoly) -> Vec<(FfPoly, usize)> {
    let mut result = Vec::new();
    let fp = f.derivative();
    if fp.is_zero() {
        return vec![(f.clone(), 1)];
    }
    let mut g = ff_gcd(f, &fp);
    let (mut s, _) = f.div_rem(&g).unwrap();
    let mut i = 1;
    while s.degree().unwrap_or(0) > 0 {
        let h = ff_gcd(&s, &g);
        let (factor, _) = s.div_rem(&h).unwrap();
        if factor.degree().unwrap_or(0) > 0 {
            result.push((factor.make_monic(), i));
        }
        s = h.clone();
        let (new_g, _) = g.div_rem(&h).unwrap();
        g = new_g;
        i += 1;
    }
    if g.degree().unwrap_or(0) > 0 {
        // Remaining repeated content; fold into the highest seen level.
        result.push((g.make_monic(), i));
    }
    if result.is_empty() {
        result.push((f.clone(), 1));
    }
    result
}

/// Test whether `F ∈ ℚ(t)[x]` is irreducible over `ℚ(t)`.
///
/// A degree-0 polynomial is not irreducible; degree-1 is irreducible. For higher
/// degree we run [`factor_over_qt`] and report a single multiplicity-one factor.
pub fn is_irreducible_over_qt(f: &FfPoly) -> Result<bool, String> {
    match f.degree() {
        None | Some(0) => Ok(false),
        Some(1) => Ok(true),
        Some(_) => {
            let (_, factors) = factor_over_qt(f);
            Ok(factors.len() == 1 && factors[0].1 == 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function_field::ff_poly_from_coeffs;
    use rustmath_core::Ring;

    fn q(n: i64) -> Rational {
        Rational::from_i64(n)
    }

    /// x^2 - t^2  ∈ ℚ(t)[x]
    fn x2_minus_t2() -> FfPoly {
        let t = RationalFunction::t();
        let minus_t2 = RationalFunction::zero() - (t.clone() * t);
        ff_poly_from_coeffs(vec![minus_t2, RationalFunction::zero(), RationalFunction::one()])
    }

    #[test]
    fn factor_x2_minus_t2() {
        let f = x2_minus_t2();
        let (content, factors) = factor_over_qt(&f);
        assert_eq!(content, RationalFunction::one());
        // Two distinct linear factors x - t and x + t, each multiplicity 1.
        assert_eq!(factors.iter().map(|(_, m)| *m).sum::<usize>(), 2);
        assert_eq!(factors.len(), 2);
        for (g, m) in &factors {
            assert_eq!(*m, 1);
            assert_eq!(g.degree(), Some(1));
        }
        // Reconstruct the product and compare to F.
        let mut prod = ff_poly_from_coeffs(vec![content]);
        for (g, m) in &factors {
            for _ in 0..*m {
                prod = prod * g.clone();
            }
        }
        assert_eq!(prod, f);
    }

    #[test]
    fn irreducible_quadratic() {
        // x^2 - t : irreducible over ℚ(t) (t is not a square in ℚ(t)).
        let t = RationalFunction::t();
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - t,
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        assert!(is_irreducible_over_qt(&f).unwrap());
        let (_, factors) = factor_over_qt(&f);
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 1);
    }

    #[test]
    fn irreducible_cubic() {
        // x^3 - t : irreducible over ℚ(t).
        let t = RationalFunction::t();
        let f = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - t,
            RationalFunction::zero(),
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        assert!(is_irreducible_over_qt(&f).unwrap());
    }

    #[test]
    fn factor_product_of_two_linears() {
        // (x - t)(x - (t+1)) = x^2 - (2t+1)x + t(t+1)
        let t = RationalFunction::t();
        let f1 = ff_poly_from_coeffs(vec![RationalFunction::zero() - t.clone(), RationalFunction::one()]);
        let f2 = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - (t.clone() + RationalFunction::one()),
            RationalFunction::one(),
        ]);
        let f = f1.clone() * f2.clone();
        let (_, factors) = factor_over_qt(&f);
        assert_eq!(factors.len(), 2);
        let total: usize = factors.iter().map(|(_, m)| *m).sum();
        assert_eq!(total, 2);
        // Each factor is linear.
        for (g, _) in &factors {
            assert_eq!(g.degree(), Some(1));
        }
    }

    #[test]
    fn factor_with_multiplicity() {
        // (x - t)^2 = x^2 - 2t x + t^2
        let t = RationalFunction::t();
        let lin = ff_poly_from_coeffs(vec![RationalFunction::zero() - t.clone(), RationalFunction::one()]);
        let f = lin.clone() * lin.clone();
        let (_, factors) = factor_over_qt(&f);
        // One distinct factor, multiplicity 2.
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].1, 2);
        assert_eq!(factors[0].0.degree(), Some(1));
    }

    #[test]
    fn reducible_quartic_into_two_quadratics() {
        // (x^2 - t)(x^2 - (t+1)): two irreducible quadratics over ℚ(t).
        let t = RationalFunction::t();
        let q1 = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - t.clone(),
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        let q2 = ff_poly_from_coeffs(vec![
            RationalFunction::zero() - (t.clone() + RationalFunction::one()),
            RationalFunction::zero(),
            RationalFunction::one(),
        ]);
        let f = q1.clone() * q2.clone();
        let (_, factors) = factor_over_qt(&f);
        let total: usize = factors.iter().map(|(g, m)| g.degree().unwrap() * m).sum();
        assert_eq!(total, 4);
        // Not irreducible.
        assert!(!is_irreducible_over_qt(&f).unwrap());
    }
}
