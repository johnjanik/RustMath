//! Absolute defining polynomial of a relative extension (note Alg 1A —
//! `docs/algorithm_notes/abext_notes.md` §1, the "multiplication-matrix
//! `rnfequation`").
//!
//! # Problem
//!
//! Given `K = ℚ[T]/(g)` with `g` monic of degree `d`, and a monic irreducible
//! relative polynomial `h(X) = Xⁿ + c_{n-1}Xⁿ⁻¹ + ⋯ + c₀ ∈ K[X]`, compute an
//! absolute polynomial `f(Y) ∈ ℚ[Y]` of degree `dn` defining
//! `L = K[X]/(h)` (Magma/PARI `rnfequation`).
//!
//! # Why a multiplication matrix and not a bivariate resultant
//!
//! The previous P2a attempt built `f` by a *bivariate resultant*
//! (`Res_T(g(T), Hₛ(T,Y))`), which expands a large symbolic elimination
//! determinant and crashed (`bivariate.rs:79`). This module instead keeps the
//! whole computation inside a **fixed `dn × dn` matrix over ℚ**: the matrix of
//! multiplication by the primitive element `ηₛ = θ + s·α` acting on the
//! `ℚ`-vector space `L` with basis `{θⁱαʲ : 0 ≤ i < d, 0 ≤ j < n}`. Its
//! characteristic polynomial is the absolute defining polynomial whenever `ηₛ`
//! is primitive.
//!
//! # Boundedness (no open-ended search)
//!
//! For all but finitely many integer shifts `s`, `ηₛ` is primitive. A collision
//! `θᵢ + s·αᵢⱼ = θᵢ′ + s·αᵢ′ⱼ′` between two of the `dn` conjugate pairs pins down
//! at most one bad `s`, so there are at most `C(dn, 2)` bad shifts and some
//! `s ∈ {0, 1, …, C(dn, 2)}` is good. We test primitivity by **separability**:
//! `ηₛ` is primitive ⟺ its `dn` conjugates are distinct ⟺ the characteristic
//! polynomial `Fₛ = det(YI − Mₛ)` is square-free (then, since `h` is irreducible
//! over `K` and `g` over `ℚ`, `Fₛ` is automatically irreducible of degree `dn`).
//! The loop is therefore a finite, deterministic scan.

use rustmath_polynomials::univariate::UnivariatePolynomial;
use rustmath_rationals::Rational;

type Q = Rational;

fn qzero() -> Q {
    Q::from_integer(0)
}

fn is_zero(x: &Q) -> bool {
    x == &qzero()
}

/// Result of the absolute-polynomial construction.
#[derive(Clone, Debug)]
pub struct AbsoluteField {
    /// Monic absolute defining polynomial `f(Y) ∈ ℚ[Y]` of degree `dn`.
    pub poly: UnivariatePolynomial<Q>,
    /// The shift `s` such that `η = θ + s·α` is the primitive element used.
    pub shift: i64,
}

/// Reduce a `θ`-polynomial `p` (coefficients low→high) modulo the monic `g`,
/// returning the remainder padded to length exactly `d = deg g`.
fn rem_g(mut p: Vec<Q>, g_full: &[Q]) -> Vec<Q> {
    let d = g_full.len() - 1; // g monic of degree d
    // Trim leading (high-degree) zeros first.
    while p.len() > d && is_zero(&p[p.len() - 1]) {
        p.pop();
    }
    while p.len() > d {
        let m = p.len() - 1; // current degree, m ≥ d
        let c = p[m].clone(); // leading coeff (g is monic, so no division)
        // p -= c · θ^{m-d} · g
        for k in 0..=d {
            p[m - d + k] = p[m - d + k].clone() - c.clone() * g_full[k].clone();
        }
        // p[m] is now exactly zero; drop it and any new trailing zeros.
        p.pop();
        while p.len() > d && is_zero(&p[p.len() - 1]) {
            p.pop();
        }
    }
    p.resize(d, qzero());
    p
}

/// Multiply two `θ`-polynomials and reduce modulo `g`. Output length `d`.
fn mul_mod_g(a: &[Q], b: &[Q], g_full: &[Q]) -> Vec<Q> {
    if a.is_empty() || b.is_empty() {
        return vec![qzero(); g_full.len() - 1];
    }
    let mut prod = vec![qzero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        if is_zero(ai) {
            continue;
        }
        for (j, bj) in b.iter().enumerate() {
            prod[i + j] = prod[i + j].clone() + ai.clone() * bj.clone();
        }
    }
    rem_g(prod, g_full)
}

/// Build the absolute defining polynomial of `L = K[X]/(h)` where
/// `K = ℚ[T]/(g)`, via Algorithm 1A (multiplication-matrix characteristic
/// polynomial of `θ + s·α`).
///
/// * `g` — monic defining polynomial of `K` over `ℚ` (degree `d ≥ 1`).
/// * `h_coeffs` — the constant→`X^{n-1}` coefficients `[c₀, c₁, …, c_{n-1}]` of
///   the **monic** relative polynomial `h(X) = Xⁿ + c_{n-1}Xⁿ⁻¹ + ⋯ + c₀`,
///   each `cₗ ∈ K` given as a `ℚ`-polynomial in `θ` (degree `< d`). `n ≥ 1`.
///
/// Returns the monic absolute polynomial `f` of degree `dn` and the shift `s`,
/// or an error if `g` is not monic / inputs are degenerate / the (finite)
/// primitive-element scan is exhausted (which cannot happen for an irreducible
/// `h`).
pub fn absolute_defining_polynomial(
    g: &UnivariatePolynomial<Q>,
    h_coeffs: &[Vec<Q>],
) -> Result<AbsoluteField, String> {
    let g_full: Vec<Q> = g.coefficients().to_vec();
    if g_full.is_empty() {
        return Err("g must be nonzero".into());
    }
    let d = g_full.len() - 1;
    if d == 0 {
        return Err("g must have degree ≥ 1 (K must be a proper field)".into());
    }
    if !is_zero(&(g_full[d].clone() - Q::from_integer(1))) {
        return Err("g must be monic".into());
    }
    let n = h_coeffs.len();
    if n == 0 {
        return Err("h must have degree ≥ 1".into());
    }
    let dim = d * n;

    // idx(i,j) = i*n + j  is the coordinate of θⁱαʲ.
    let idx = |i: usize, j: usize| i * n + j;

    // Precompute θ⁰,…,θ^d reduced mod g (we need up to θ^d when i = d-1).
    let theta_pow: Vec<Vec<Q>> = (0..=d)
        .map(|e| {
            let mut v = vec![qzero(); e + 1];
            v[e] = Q::from_integer(1);
            rem_g(v, &g_full)
        })
        .collect();

    // h coefficients cₗ reduced mod g (each length d). α^n = −Σ_l cₗ α^l.
    let c: Vec<Vec<Q>> = h_coeffs.iter().map(|cl| rem_g(cl.clone(), &g_full)).collect();

    // Columns of the multiplication-by-θ matrix T and multiplication-by-α
    // matrix A, expressed on the basis {θⁱαʲ}. (Charpoly is transpose-invariant,
    // so the row/column convention is immaterial.)
    let mut t_cols = vec![vec![qzero(); dim]; dim];
    let mut a_cols = vec![vec![qzero(); dim]; dim];

    for i in 0..d {
        for j in 0..n {
            let b = idx(i, j); // basis element θⁱαʲ

            // θ · θⁱαʲ = θ^{i+1} αʲ ; θ^{i+1} = Σ_k theta_pow[i+1][k] θ^k.
            let tp = &theta_pow[i + 1];
            for (k, tk) in tp.iter().enumerate() {
                if !is_zero(tk) {
                    t_cols[b][idx(k, j)] = t_cols[b][idx(k, j)].clone() + tk.clone();
                }
            }

            // α · θⁱαʲ = θⁱ α^{j+1}.
            if j + 1 < n {
                a_cols[b][idx(i, j + 1)] = Q::from_integer(1);
            } else {
                // α^n = −Σ_l cₗ α^l, so θⁱ·α^n = −Σ_l (θⁱ·cₗ) α^l.
                for (l, cl) in c.iter().enumerate() {
                    let prod = mul_mod_g(&theta_pow[i], cl, &g_full);
                    for (k, pk) in prod.iter().enumerate() {
                        if !is_zero(pk) {
                            a_cols[b][idx(k, l)] =
                                a_cols[b][idx(k, l)].clone() - pk.clone();
                        }
                    }
                }
            }
        }
    }

    // Finite bad-s bound: at most C(dim,2) bad shifts.
    let bound: i64 = (dim as i64) * (dim as i64 - 1) / 2;

    for s in 0..=bound {
        let sq = Q::from_integer(s);
        // Mₛ = T + s·A as a flat dim×dim matrix (row r, col b).
        let mut data = Vec::with_capacity(dim * dim);
        for r in 0..dim {
            for b in 0..dim {
                data.push(t_cols[b][r].clone() + sq.clone() * a_cols[b][r].clone());
            }
        }
        let m = rustmath_matrix::matrix::Matrix::from_vec(dim, dim, data)
            .map_err(|e| format!("matrix build failed: {e:?}"))?;
        let f = rustmath_matrix::companion::characteristic_polynomial(&m)
            .map_err(|e| format!("charpoly failed: {e:?}"))?;
        // Primitive ⟺ Fₛ square-free (distinct conjugates). Degree is always dim.
        if f.degree() == Some(dim) && f.is_square_free() {
            return Ok(AbsoluteField { poly: f, shift: s });
        }
    }

    Err(format!(
        "primitive-element scan exhausted at s={bound} (h not irreducible over K?)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qi(n: i64) -> Q {
        Q::from_integer(n)
    }

    /// Build a `UnivariatePolynomial<Q>` from integer coefficients low→high.
    fn upoly(coeffs: &[i64]) -> UnivariatePolynomial<Q> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| qi(c)).collect())
    }

    #[test]
    fn theta_reduction_basic() {
        // g = T² + 23, so θ² = −23.
        let g = vec![qi(23), qi(0), qi(1)];
        let red = rem_g(vec![qi(0), qi(0), qi(1)], &g); // θ²
        assert_eq!(red, vec![qi(-23), qi(0)]);
        // θ³ = −23 θ
        let red3 = rem_g(vec![qi(0), qi(0), qi(0), qi(1)], &g);
        assert_eq!(red3, vec![qi(0), qi(-23)]);
    }

    #[test]
    fn hcf_qsqrtm23_absolute_polynomial() {
        // K = ℚ(√−23): g = T² + 23. Relative cubic h = X³ − X − 1 over K.
        // The note (validated against PARI `rnfequation`) gives, for y = θ + α
        // (shift s = 1), the absolute polynomial
        //     y⁶ + 67y⁴ − 2y³ + 1588y² + 140y + 13249.
        let g = upoly(&[23, 0, 1]);
        // h = X³ + 0·X² + (−1)·X + (−1); coeffs c₀=−1, c₁=−1, c₂=0, each in K.
        let h_coeffs = vec![vec![qi(-1)], vec![qi(-1)], vec![qi(0)]];

        let res = absolute_defining_polynomial(&g, &h_coeffs).expect("construction");
        // The smallest good shift is s = 1 (s = 0 gives η = θ, generating only K,
        // whose charpoly (Y²+23)³ is not square-free).
        assert_eq!(res.shift, 1, "expected smallest good shift s=1");

        let expected = upoly(&[13249, 140, 1588, -2, 67, 0, 1]);
        assert_eq!(
            res.poly, expected,
            "absolute polynomial must match the PARI-validated note value"
        );
        assert_eq!(res.poly.degree(), Some(6));
        assert!(res.poly.is_square_free());
    }

    #[test]
    fn shift_zero_is_rejected_not_primitive() {
        // η = θ alone is not primitive for L/ℚ: its charpoly is (Y²+23)³,
        // degree 6 but not square-free, so s=0 must be skipped.
        let g = upoly(&[23, 0, 1]);
        let h_coeffs = vec![vec![qi(-1)], vec![qi(-1)], vec![qi(0)]];
        let res = absolute_defining_polynomial(&g, &h_coeffs).unwrap();
        assert!(res.shift >= 1);
    }

    #[test]
    fn degenerate_base_k_equals_q() {
        // g = T gives K = ℚ (θ = 0). The relative X² − 2 over ℚ must yield the
        // absolute polynomial Y² − 2 (degree dn = 2), shift s = 1.
        let g = upoly(&[0, 1]); // T  (degree 1)
        let h_coeffs = vec![vec![qi(-2)], vec![qi(0)]]; // X² − 2
        let res = absolute_defining_polynomial(&g, &h_coeffs).unwrap();
        assert_eq!(res.poly, upoly(&[-2, 0, 1]));
        assert_eq!(res.shift, 1);
    }

    #[test]
    fn constant_g_is_rejected() {
        // g of degree 0 is the zero ring — must error.
        let g = upoly(&[5]);
        let h_coeffs = vec![vec![qi(-2)], vec![qi(0)]];
        assert!(absolute_defining_polynomial(&g, &h_coeffs).is_err());
    }
}
