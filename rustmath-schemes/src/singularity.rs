//! Deciding smoothness: the Jacobian criterion, made into a machine-checkable test.
//!
//! # What is actually decided here
//!
//! Everything rests on one unconditional fact:
//!
//! > For an ideal `I ⊆ k[x₀,…,x_{n−1}]` with `k` any field, `1 ∈ I` **iff** the system
//! > `g = 0 (g ∈ I)` has no solution in `k̄ⁿ`.
//!
//! (⇐ is the weak Nullstellensatz; ⇒ is trivial.) Crucially, `1 ∈ I` is decided by a
//! Gröbner basis computed **over `k` itself**: Buchberger's algorithm performs only field
//! operations on the coefficients it is given, so a Gröbner basis of `I` over `k` is still
//! a Gröbner basis of `I·k̄[x]` over `k̄`. So computing over ℚ genuinely settles the
//! question over ℚ̄ — no extension arithmetic and no overclaiming.
//!
//! [`is_unit_ideal`] is that decision procedure; the rest of the module builds the right
//! ideal to feed it.
//!
//! # The hypersurface case is hypothesis-free
//!
//! For a single `f ∈ k[x₀,…,x_{n−1}]`, put
//!
//! ```text
//! Sing(f) = ( f, ∂f/∂x₀, …, ∂f/∂x_{n−1} ).
//! ```
//!
//! Then, over a field of characteristic 0:
//!
//! - `1 ∈ Sing(f)` ⟺ no point of `k̄ⁿ` kills `f` and all its partials
//!   ⟺ **`V(f)` is a smooth hypersurface over `k̄`** (at every point of `V(f)` some partial
//!   is non-zero, so the Jacobian has rank 1 = codim).
//! - This *also* forces `f` to be squarefree: if `f = g²·h` then `f` and every `∂f/∂x_i`
//!   vanish on `V(g)`, which is non-empty over `k̄`, so `1 ∉ Sing(f)`.
//!
//! So for a hypersurface the unit-ideal test needs **no hypotheses at all** — not
//! reducedness, not irreducibility. That is why [`is_smooth_hypersurface`] is the gate the
//! blow-up tests use. The converse direction is equally unconditional: `1 ∉ Sing(f)` gives
//! a point of `k̄ⁿ` where `f` and all partials vanish, so `V(f)` is not smooth over `k̄`
//! (it is singular, or non-reduced, there).
//!
//! # The general case has real hypotheses, and we state them
//!
//! For an ideal with several generators the criterion needs the *codimension*, and it is
//! only valid when `V(I)` is equidimensional (see [`singular_subscheme`]). We do not
//! silently pretend otherwise: that function documents the hypothesis, and callers who
//! want an unconditional answer should use the hypersurface entry point.

use rustmath_core::Field;
use rustmath_polynomials::elimination::krull_dimension;
use rustmath_polynomials::groebner::{groebner_basis_field, GroebnerBudget, MonomialOrdering};
use rustmath_polynomials::multivariate::MultivariatePolynomial;

/// Decide whether `1 ∈ I`, i.e. whether `V(I) = ∅` over the algebraic closure of `k`.
///
/// Computes a Gröbner basis and checks it for a non-zero constant. A non-zero constant in
/// the basis generates the unit ideal; conversely, if `1 ∈ I` then `1` reduces to `0`
/// modulo any Gröbner basis of `I`, which forces some basis element to have a monomial `1`
/// as its leading monomial — i.e. to be a non-zero constant. So the test is exact in both
/// directions, not a heuristic.
///
/// By the base-change remark in the module docs, the answer computed over `k` is the answer
/// over `k̄`.
pub fn is_unit_ideal<R: Field>(
    gens: Vec<MultivariatePolynomial<R>>,
    budget: &GroebnerBudget,
) -> Result<bool, String> {
    let gens: Vec<_> = gens.into_iter().filter(|g| !g.is_zero()).collect();
    if gens.is_empty() {
        return Ok(false); // the zero ideal
    }

    // A non-zero constant among the generators already settles it.
    if gens.iter().any(|g| g.is_constant() && !g.is_zero()) {
        return Ok(true);
    }

    let gb = groebner_basis_field(gens, MonomialOrdering::Grevlex, budget)?;
    Ok(gb.iter().any(|g| !g.is_zero() && g.is_constant()))
}

/// The Jacobian matrix `(∂gᵢ/∂xⱼ)` of `gens` in `n` variables: `gens.len()` rows, `n` columns.
pub fn jacobian<R: Field>(
    gens: &[MultivariatePolynomial<R>],
    n: usize,
) -> Vec<Vec<MultivariatePolynomial<R>>> {
    gens.iter()
        .map(|g| (0..n).map(|j| g.partial_derivative(j)).collect())
        .collect()
}

/// The ideal `Sing(f) = (f, ∂f/∂x₀, …, ∂f/∂x_{n−1})` cutting out the non-smooth locus of
/// the hypersurface `V(f) ⊆ 𝔸ⁿ`.
///
/// This is exactly the singular subscheme of `V(f)`; see the module docs for why it needs
/// no hypotheses on `f`.
pub fn hypersurface_singular_locus<R: Field>(
    f: &MultivariatePolynomial<R>,
    n: usize,
) -> Vec<MultivariatePolynomial<R>> {
    let mut gens = vec![f.clone()];
    for j in 0..n {
        gens.push(f.partial_derivative(j));
    }
    gens
}

/// Is the hypersurface `V(f) ⊆ 𝔸ⁿ` smooth over `k̄`?
///
/// **Asserts precisely**: there is no point of `k̄ⁿ` at which `f` and all of its `n` partial
/// derivatives vanish simultaneously. Over a characteristic-0 field that is equivalent to
/// `V(f)` being a smooth (in particular reduced) hypersurface over `k̄`.
///
/// Errors — rather than guessing — if `f` is zero (`V(0) = 𝔸ⁿ` is smooth but is not a
/// hypersurface, and calling it one would be a category error) or if `f` is a non-zero
/// constant (`V(f) = ∅`, again not a hypersurface).
pub fn is_smooth_hypersurface<R: Field>(
    f: &MultivariatePolynomial<R>,
    n: usize,
    budget: &GroebnerBudget,
) -> Result<bool, String> {
    if f.is_zero() {
        return Err("is_smooth_hypersurface: f = 0 does not define a hypersurface".to_string());
    }
    if f.is_constant() {
        return Err(format!(
            "is_smooth_hypersurface: f = {} is a non-zero constant, so V(f) = ∅ is not a \
             hypersurface",
            f
        ));
    }
    if let Some(v) = f.max_variable() {
        if v >= n {
            return Err(format!(
                "is_smooth_hypersurface: f uses variable x{} but n = {}",
                v, n
            ));
        }
    }
    is_unit_ideal(hypersurface_singular_locus(f, n), budget)
}

/// The `c × c` minors of the Jacobian, where `c` is the codimension.
///
/// Refuses combinatorially unreasonable requests rather than grinding.
fn jacobian_minors<R: Field>(
    gens: &[MultivariatePolynomial<R>],
    n: usize,
    c: usize,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    let r = gens.len();
    if c == 0 || c > r || c > n {
        return Err(format!(
            "jacobian_minors: cannot take {}×{} minors of a {}×{} Jacobian",
            c, c, r, n
        ));
    }

    let count = binomial(r, c).saturating_mul(binomial(n, c));
    if count > 20_000 {
        return Err(format!(
            "jacobian_minors: {} minors of size {} is too many; refusing rather than grinding",
            count, c
        ));
    }

    let jac = jacobian(gens, n);
    let mut minors = Vec::new();
    for rows in combinations(r, c) {
        for cols in combinations(n, c) {
            let sub: Vec<Vec<MultivariatePolynomial<R>>> = rows
                .iter()
                .map(|&i| cols.iter().map(|&j| jac[i][j].clone()).collect())
                .collect();
            let d = determinant(&sub);
            if !d.is_zero() {
                minors.push(d);
            }
        }
    }
    Ok(minors)
}

/// Laplace expansion. `c` is tiny here (the codimension), so this is fine.
fn determinant<R: Field>(m: &[Vec<MultivariatePolynomial<R>>]) -> MultivariatePolynomial<R> {
    let c = m.len();
    if c == 1 {
        return m[0][0].clone();
    }
    let mut det = MultivariatePolynomial::zero();
    for j in 0..c {
        let minor: Vec<Vec<MultivariatePolynomial<R>>> = m[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(k, _)| *k != j)
                    .map(|(_, e)| e.clone())
                    .collect()
            })
            .collect();
        let term = m[0][j].clone() * determinant(&minor);
        det = if j % 2 == 0 { det + term } else { det - term };
    }
    det
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut r: usize = 1;
    for i in 0..k {
        r = r.saturating_mul(n - i) / (i + 1);
    }
    r
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..k).collect();
    if k > n {
        return out;
    }
    loop {
        out.push(idx.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

/// The singular subscheme of `V(I) ⊆ 𝔸ⁿ`, as `I + I_c(Jac)` with `c = n − dim V(I)`.
///
/// # Hypotheses — read these
///
/// The Jacobian criterion in this `c × c`-minors form computes the non-smooth locus of
/// `V(I)` **only when `V(I)` is equidimensional of codimension `c`** (all components of the
/// same dimension) over a perfect field. `c` is derived here from the Krull dimension, so
/// it is the codimension of the *largest* component; if `V(I)` has components of different
/// dimensions, the ideal returned is not the singular subscheme and this function's answer
/// is not meaningful. We cannot check equidimensionality without primary decomposition,
/// which this crate does not have — so this is a documented precondition on the caller, not
/// something silently assumed to hold.
///
/// For a hypersurface use [`hypersurface_singular_locus`] instead: it is unconditional.
///
/// Returns `Err` when `V(I)` is empty (`c` is not defined) or the ambient is the whole
/// space (`I = (0)`, which is smooth but not cut out by anything).
pub fn singular_subscheme<R: Field>(
    gens: Vec<MultivariatePolynomial<R>>,
    n: usize,
    budget: &GroebnerBudget,
) -> Result<Vec<MultivariatePolynomial<R>>, String> {
    let gens: Vec<_> = gens.into_iter().filter(|g| !g.is_zero()).collect();
    if gens.is_empty() {
        return Err("singular_subscheme: I = (0); V(I) = 𝔸ⁿ is smooth, but it is not cut \
                    out by any equations, so there is no Jacobian to take minors of"
            .to_string());
    }

    let dim = krull_dimension(gens.clone(), n, budget)?;
    if dim < 0 {
        return Err("singular_subscheme: V(I) = ∅, so its codimension is undefined".to_string());
    }
    let c = n - dim as usize;

    let mut result = gens.clone();
    result.extend(jacobian_minors(&gens, n, c)?);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustmath_integers::Integer;
    use rustmath_polynomials::multivariate::Monomial;
    use rustmath_rationals::Rational;

    type P = MultivariatePolynomial<Rational>;

    fn q(n: i64) -> Rational {
        Rational::from_integer(Integer::from(n))
    }

    fn term(c: i64, powers: &[(usize, u32)]) -> P {
        let mut m = Monomial::new();
        for (v, e) in powers {
            m = m.mul(&Monomial::variable(*v, *e));
        }
        let mut p = P::zero();
        p.add_term(m, q(c));
        p
    }

    fn con(c: i64) -> P {
        term(c, &[])
    }

    fn budget() -> GroebnerBudget {
        GroebnerBudget::generous()
    }

    #[test]
    fn unit_ideal_detection_is_exact() {
        // (x, 1 - x) = (1): the two generators sum to 1.
        let gens = vec![term(1, &[(0, 1)]), con(1) - term(1, &[(0, 1)])];
        assert!(is_unit_ideal(gens, &budget()).unwrap());

        // (x, y) is the origin — a real point, not the unit ideal.
        let gens = vec![term(1, &[(0, 1)]), term(1, &[(1, 1)])];
        assert!(!is_unit_ideal(gens, &budget()).unwrap());

        // The zero ideal is not the unit ideal.
        assert!(!is_unit_ideal(Vec::<P>::new(), &budget()).unwrap());

        // A bare non-zero constant is.
        assert!(is_unit_ideal(vec![con(7)], &budget()).unwrap());

        // (x^2 + 1) has no *rational* root, but it has complex roots — it must NOT be
        // called the unit ideal. This is the test that catches a "no solutions over ℚ"
        // confusion masquerading as emptiness.
        let x2_plus_1 = term(1, &[(0, 2)]) + con(1);
        assert!(!is_unit_ideal(vec![x2_plus_1], &budget()).unwrap());
    }

    #[test]
    fn smooth_conic_is_smooth_singular_conic_is_not() {
        // x^2 + y^2 - 1: smooth circle. Sing = (f, 2x, 2y) = (x, y, -1) ∋ 1.
        let circle = term(1, &[(0, 2)]) + term(1, &[(1, 2)]) - con(1);
        assert!(is_smooth_hypersurface(&circle, 2, &budget()).unwrap());

        // x^2 - y^2: two crossing lines, singular at the origin.
        let cross = term(1, &[(0, 2)]) - term(1, &[(1, 2)]);
        assert!(!is_smooth_hypersurface(&cross, 2, &budget()).unwrap());

        // x^2: non-reduced. The unit-ideal test must reject it, as promised in the docs
        // (f = g^2 makes f and all its partials vanish on V(g)).
        let double_line = term(1, &[(0, 2)]);
        assert!(!is_smooth_hypersurface(&double_line, 2, &budget()).unwrap());

        // A line is smooth.
        let line = term(1, &[(0, 1)]) - term(1, &[(1, 1)]);
        assert!(is_smooth_hypersurface(&line, 2, &budget()).unwrap());
    }

    #[test]
    fn hypersurface_entry_point_refuses_non_hypersurfaces() {
        assert!(is_smooth_hypersurface(&P::zero(), 2, &budget()).is_err());
        assert!(is_smooth_hypersurface(&con(1), 2, &budget()).is_err());
    }

    #[test]
    fn general_singular_subscheme_agrees_with_the_hypersurface_route() {
        // For a hypersurface, codim = 1 and the 1x1 minors are exactly the partials, so
        // the general routine must reproduce the unconditional one.
        let cusp = term(1, &[(1, 2)]) - term(1, &[(0, 3)]); // y^2 - x^3
        let general = singular_subscheme(vec![cusp.clone()], 2, &budget()).unwrap();
        let special = hypersurface_singular_locus(&cusp, 2);

        // Both must be non-unit (the cusp is singular) and cut out the same locus.
        assert!(!is_unit_ideal(general.clone(), &budget()).unwrap());
        assert!(!is_unit_ideal(special.clone(), &budget()).unwrap());

        // Mutual membership: same ideal.
        use rustmath_polynomials::groebner::ideal_membership_field;
        for g in &general {
            assert!(
                ideal_membership_field(g, &special, MonomialOrdering::Grevlex, &budget()).unwrap()
            );
        }
        for s in &special {
            assert!(
                ideal_membership_field(s, &general, MonomialOrdering::Grevlex, &budget()).unwrap()
            );
        }
    }

    #[test]
    fn singular_subscheme_of_the_twisted_cubic_is_empty() {
        // The twisted cubic (y - x^2, z - x^3) is a smooth curve in A^3: codim 2, and the
        // 2x2 minors of the Jacobian never all vanish on it. Derived independently in
        // sympy; here the assertion is that the singular subscheme is the unit ideal.
        // x = 0, y = 1, z = 2
        let tc = vec![
            term(1, &[(1, 1)]) - term(1, &[(0, 2)]),
            term(1, &[(2, 1)]) - term(1, &[(0, 3)]),
        ];
        let sing = singular_subscheme(tc, 3, &budget()).unwrap();
        assert!(
            is_unit_ideal(sing, &budget()).unwrap(),
            "the twisted cubic is smooth"
        );
    }

    #[test]
    fn singular_subscheme_refuses_the_empty_and_the_zero_ideal() {
        assert!(singular_subscheme(Vec::<P>::new(), 2, &budget()).is_err());
        assert!(singular_subscheme(vec![con(1)], 2, &budget()).is_err());
    }

    #[test]
    fn combinations_enumerates_correctly() {
        assert_eq!(combinations(4, 2).len(), 6);
        assert_eq!(combinations(3, 1), vec![vec![0], vec![1], vec![2]]);
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(2, 5), 0);
    }
}
