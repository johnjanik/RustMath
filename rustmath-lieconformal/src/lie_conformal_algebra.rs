//! Base Lie Conformal Algebra Module
//!
//! Defines the fundamental traits and structures for Lie conformal algebras.

use rustmath_core::Ring;
use std::fmt::{self, Display};
use std::collections::HashMap;
use crate::lie_conformal_algebra_element::LieConformalAlgebraElement;

/// Trait for Lie conformal algebras
pub trait LieConformalAlgebra<R: Ring> {
    type Element;
    fn base_ring(&self) -> &R;
    fn ngens(&self) -> Option<usize>;
    fn generator(&self, i: usize) -> Option<Self::Element>;
    fn generators(&self) -> Vec<Self::Element> {
        if let Some(n) = self.ngens() {
            (0..n).filter_map(|i| self.generator(i)).collect()
        } else {
            vec![]
        }
    }
    fn is_abelian(&self) -> bool {
        false
    }
    fn central_charge(&self) -> Option<R> {
        None
    }
    fn zero(&self) -> Self::Element;
    fn one(&self) -> Option<Self::Element> {
        None
    }
}

/// λ-bracket operation for Lie conformal algebras
///
/// `lambda_bracket(a, b)` returns the j-th products `a_(j) b` keyed by `j`,
/// where `[a_λ b] = Σ_j λ^j / j! · a_(j) b` (Kac, *Vertex Algebras for
/// Beginners*, §2.7; this is also SageMath's `bracket` convention). Keys whose
/// product is zero are omitted, so an empty map means `[a_λ b] = 0`.
pub trait LambdaBracket<R: Ring, E: Clone> {
    fn lambda_bracket(&self, a: &E, b: &E) -> HashMap<usize, E>;
    fn n_product(&self, a: &E, b: &E, n: usize) -> E {
        self.lambda_bracket(a, b).get(&n).cloned()
            .unwrap_or_else(|| panic!("No term for power {}", n))
    }
}

/// Sesquilinear extension of a λ-bracket whose value on two generators is a
/// multiple of a central element `K` with `∂K = 0`.
///
/// Suppose `[a_λ b] = λ^m · K` for generators `a`, `b` (the caller multiplies
/// by the structure constant afterwards). Sesquilinearity
/// (`[∂a_λ b] = -λ[a_λ b]`, `[a_λ ∂b] = (∂ + λ)[a_λ b]`, Kac §2.7) gives
///
/// `[p(∂)a_λ q(∂)b] = p(-λ) q(∂ + λ) λ^m K = p(-λ) q(λ) λ^m K`,
///
/// the last step because `∂K = 0`. Here `p` and `q` are coefficient vectors in
/// ordinary powers of `∂` (index `i` holds the coefficient of `∂^i`), as in
/// [`crate::LieConformalAlgebraElement`].
///
/// Returns `v` with `v[j]` the coefficient of `K` in the j-th product
/// `(p(∂)a)_(j)(q(∂)b) = j! · [λ^j]`.
pub(crate) fn central_sesquilinear<R: Ring + From<i64>>(p: &[R], q: &[R], m: usize) -> Vec<R> {
    if p.is_empty() || q.is_empty() {
        return Vec::new();
    }
    // p(-λ) q(λ): the coefficient of λ^(i+k) gets (-1)^i p_i q_k.
    let mut prod = vec![R::zero(); p.len() + q.len() - 1 + m];
    for (i, pi) in p.iter().enumerate() {
        if pi.is_zero() {
            continue;
        }
        let signed = if i % 2 == 0 { pi.clone() } else { -pi.clone() };
        for (k, qk) in q.iter().enumerate() {
            if qk.is_zero() {
                continue;
            }
            let idx = i + k + m;
            prod[idx] = prod[idx].clone() + signed.clone() * qk.clone();
        }
    }
    // Convert the coefficient of λ^j to the j-th product: multiply by j!.
    let mut factorial = R::from(1);
    for (j, c) in prod.iter_mut().enumerate() {
        if j > 0 {
            factorial = factorial * R::from(j as i64);
        }
        *c = c.clone() * factorial.clone();
    }
    prod
}

/// λ-bracket of two elements of an algebra whose generator brackets are all
/// of the form `[g_i λ g_j] = c_ij λ^(m_ij) K` with `K` central and `∂K = 0`
/// (free bosons, free fermions, bosonic and fermionic ghosts).
///
/// `pair(i, j)` returns `Some((c_ij, m_ij))`, or `None` when the bracket of
/// the two generators vanishes (in particular whenever one of them is `K`).
/// The result maps `j` to the j-th product, a multiple of `K`; zero products
/// are omitted.
///
/// # Panics
///
/// If an element has a term whose basis index is not one of the `ngens`
/// generators of the algebra.
pub(crate) fn central_valued_bracket<R, F>(
    a: &LieConformalAlgebraElement<R, GeneratorIndex>,
    b: &LieConformalAlgebraElement<R, GeneratorIndex>,
    ngens: usize,
    central_index: usize,
    pair: F,
) -> HashMap<usize, LieConformalAlgebraElement<R, GeneratorIndex>>
where
    R: Ring + From<i64>,
    F: Fn(usize, usize) -> Option<(R, usize)>,
{
    let index = |g: &GeneratorIndex| -> usize {
        match g.as_finite() {
            Some(i) if i < ngens => i,
            _ => panic!("basis index {} is not a generator of this algebra", g),
        }
    };

    let mut coeffs: Vec<R> = Vec::new();
    for (ga, p) in a.terms() {
        let i = index(ga);
        for (gb, q) in b.terms() {
            let j = index(gb);
            let Some((c, m)) = pair(i, j) else { continue };
            if c.is_zero() {
                continue;
            }
            for (n, v) in central_sesquilinear(p, q, m).into_iter().enumerate() {
                if n >= coeffs.len() {
                    coeffs.resize(n + 1, R::zero());
                }
                coeffs[n] = coeffs[n].clone() + c.clone() * v;
            }
        }
    }

    coeffs
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(n, c)| {
            (
                n,
                LieConformalAlgebraElement::with_polynomial(
                    GeneratorIndex::finite(central_index),
                    vec![c],
                ),
            )
        })
        .collect()
}

/// Derivation operator ∂ for Lie conformal algebras
pub trait Derivation<E: Clone> {
    fn apply_derivation(&self, element: &E) -> E;
    fn apply_derivation_n(&self, element: &E, n: usize) -> E {
        let mut result = element.clone();
        for _ in 0..n {
            result = self.apply_derivation(&result);
        }
        result
    }
}

/// Index set for generators
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeneratorIndex {
    Finite(usize),
    Named(String),
    Composite(Vec<Box<GeneratorIndex>>),
}

impl Display for GeneratorIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GeneratorIndex::Finite(i) => write!(f, "{}", i),
            GeneratorIndex::Named(s) => write!(f, "{}", s),
            GeneratorIndex::Composite(indices) => {
                write!(f, "(")?;
                for (i, idx) in indices.iter().enumerate() {
                    if i > 0 { write!(f, ",")?; }
                    write!(f, "{}", idx)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl GeneratorIndex {
    pub fn finite(i: usize) -> Self {
        GeneratorIndex::Finite(i)
    }
    pub fn named(s: impl Into<String>) -> Self {
        GeneratorIndex::Named(s.into())
    }
    pub fn is_finite(&self) -> bool {
        matches!(self, GeneratorIndex::Finite(_))
    }
    pub fn as_finite(&self) -> Option<usize> {
        match self {
            GeneratorIndex::Finite(i) => Some(*i),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference identities (Kac, Vertex Algebras for Beginners, (2.7.4)):
    //   (∂a)_(n) b = -n a_(n-1) b,   a_(n) (∂b) = ∂(a_(n) b) + n a_(n-1) b.
    // With [a_λ b] = λK (so a_(1)b = K, all others 0) and ∂K = 0 they give the
    // values asserted below.

    #[test]
    fn central_sesquilinear_generators() {
        // [a_λ b] = λK: only the 1st product, equal to K.
        assert_eq!(central_sesquilinear::<i64>(&[1], &[1], 1), vec![0, 1]);
        // [a_λ b] = K: only the 0th product.
        assert_eq!(central_sesquilinear::<i64>(&[1], &[1], 0), vec![1]);
    }

    #[test]
    fn central_sesquilinear_scalars() {
        // [(2a)_λ (3b)] = 6λK.
        assert_eq!(central_sesquilinear::<i64>(&[2], &[3], 1), vec![0, 6]);
    }

    #[test]
    fn central_sesquilinear_derivatives() {
        // (∂a)_(2) b = -2 a_(1) b = -2K; every other product vanishes.
        assert_eq!(central_sesquilinear::<i64>(&[0, 1], &[1], 1), vec![0, 0, -2]);
        // a_(2) (∂b) = ∂(a_(2) b) + 2 a_(1) b = 2K.
        assert_eq!(central_sesquilinear::<i64>(&[1], &[0, 1], 1), vec![0, 0, 2]);
        // (∂a)_(3)(∂b) = -3 a_(2)(∂b) = -6K, from the two identities above.
        assert_eq!(central_sesquilinear::<i64>(&[0, 1], &[0, 1], 1), vec![0, 0, 0, -6]);
    }
}
