//! # Hecke-eigenspace (newform-style) decomposition of the cuspidal space over Q
//!
//! Decomposes the cuspidal subspace of M_2(Gamma0(N)) (weight 2, trivial
//! character, sign 0) into Q-irreducible Hecke-stable subspaces:
//!
//! * the characteristic polynomial of T_p on the cuspidal subspace is
//!   factored over Q (integer factorization via Zassenhaus after clearing
//!   denominators; EVERY consumed factorization is self-certified by
//!   re-multiplying the factors, over Z and again over Q, and comparing with
//!   the input exactly);
//! * the space splits into the generalized eigenspaces ker f(T_p)^m of the
//!   irreducible factors f (primary decomposition, so no semisimplicity
//!   assumption is needed);
//! * the splitting is repeated across small primes p coprime to N until it
//!   is stable for three consecutive primes (each summand then has every
//!   examined T_p acting with a single irreducible characteristic factor).
//!
//! Multiplicities are structural, not accidental:
//!
//! * sign-0 modular symbols contain each Hecke eigensystem TWICE (the +/-
//!   pair swapped by the star involution), so a rational newform gives a
//!   2-dimensional summand with charpoly(T_p) = (x - a_p)^2;
//! * a newform with coefficient field of degree d > 1 gives a 2d-dimensional
//!   summand whose T_p-charpoly is (irreducible of degree <= d)^(2d/deg);
//!   the eigenvalue IS the irreducible polynomial, represented honestly by
//!   [`HeckeEigenvalue::Algebraic`], never coerced to a rational number;
//! * a newform of level M | N, M < N contributes sigma_0(N/M) degeneracy
//!   images that share ONE eigensystem away from N, so the T_p-refinement
//!   (p coprime to N) keeps them together in a single
//!   2 * sigma_0(N/M) * d-dimensional summand.  For gcd(n, N) > 1 the
//!   operator U_n can then act with several distinct irreducible factors,
//!   reported honestly as [`SummandHeckeAction::Mixed`].  The degeneracy
//!   maps between levels ARE now implemented ([`super::degeneracy`]); the
//!   raising maps exhibit such a block as a direct sum of copies of the
//!   lower level and reduce its U_p action to the explicit block matrices
//!   over the lower-level eigenvalues (demonstrated in the degeneracy
//!   tests at N = 22 and 44), while the summand itself is kept whole here
//!   because away from N it is a single indivisible eigensystem.
//!
//! Corresponds to `sage.modular.modsym.ambient.ModularSymbolsAmbient.decomposition`
//! and the MAGMA handbook chapter "Modular Symbols" (`NewformDecomposition`,
//! up to the old/new refinement deferred above).  References: Stein,
//! *Modular Forms: A Computational Approach*, ch. 9; Cremona, *Algorithms
//! for Modular Elliptic Curves*, ch. 2.
//!
//! Every rational eigenvalue asserted in the tests was recomputed
//! independently BEFORE implementation: conductors of the elliptic-curve
//! models from the Delta/c4 semistable criterion, a_p by direct point
//! counting over F_p (nonsingular points at multiplicative primes), U_p
//! actions on old subspaces from the explicit degeneracy-matrix theory
//! applied to those point counts.  The level-23 quadratic eigenvalue
//! polynomials were cross-checked by factoring the computed charpolys
//! independently in sympy and by the internal-consistency batteries
//! (irreducibility certificates, trace/norm integrality, discriminant in
//! Q(sqrt(5)), and the Ramanujan-Petersson bound in both real embeddings).

use super::gamma0::ModularSymbolsGamma0;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::{charpoly_berkowitz, Matrix};
use rustmath_polynomials::{zassenhaus, zx, UnivariatePolynomial};
use rustmath_rationals::Rational;

/// The eigenvalue of a Hecke operator on a Q-irreducible summand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeckeEigenvalue {
    /// The operator acts with the single rational eigenvalue a_n
    /// (charpoly = (x - a_n)^dim on the summand).
    Rational(Rational),
    /// The eigenvalue generates a field of degree > 1: it "is" this monic
    /// irreducible polynomial over Q (coefficients in increasing degree
    /// order), of which the actual eigenvalues are the conjugate roots.
    Algebraic(UnivariatePolynomial<Rational>),
}

/// How a Hecke operator T_n acts on a summand of the decomposition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SummandHeckeAction {
    /// charpoly(T_n | summand) is a power of a single irreducible
    /// polynomial: the summand carries one eigensystem for T_n.
    Eigenvalue(HeckeEigenvalue),
    /// charpoly(T_n | summand) has several distinct irreducible factors,
    /// given here with multiplicities.  This happens only for gcd(n, N) > 1
    /// on summands assembled from several degeneracy images of one newform
    /// of lower level; the degeneracy maps of [`super::degeneracy`] exhibit
    /// the block structure behind these factors explicitly.
    Mixed(Vec<(UnivariatePolynomial<Rational>, u32)>),
}

/// A Hecke-stable summand of the cuspidal subspace, Q-irreducible under the
/// Hecke operators T_p with p coprime to the level that were used in the
/// splitting.
#[derive(Debug, Clone)]
pub struct HeckeSummand {
    /// Basis vectors in the coordinates of the ambient quotient basis of
    /// M_2(Gamma0(N)) (same convention as
    /// [`ModularSymbolsGamma0::cuspidal_basis`]).
    ambient_basis: Vec<Vec<Rational>>,
    /// The same vectors in the coordinates of the cuspidal basis
    /// (coefficient vectors with respect to `cuspidal_basis()`).
    cuspidal_coords: Vec<Vec<Rational>>,
}

impl HeckeSummand {
    /// Build a summand from its two coordinate presentations (used by the
    /// U_l refinement of [`super::degeneracy`], which carves finer
    /// Hecke-stable pieces out of an existing summand).
    ///
    /// The two presentations describe the SAME list of vectors, so they must have
    /// the same length; [`Self::dimension`] reads that length off
    /// `cuspidal_coords` while [`Self::ambient_basis`] hands out the other, and a
    /// mismatch would make the summand silently self-inconsistent.  The check is
    /// a real `assert_eq!` rather than a `debug_assert_eq!` on purpose: a
    /// debug-only gate is compiled out of release builds, i.e. it is no gate at
    /// all exactly where a malformed summand would do its damage.
    pub(crate) fn new(ambient_basis: Vec<Vec<Rational>>, cuspidal_coords: Vec<Vec<Rational>>) -> Self {
        assert_eq!(
            ambient_basis.len(),
            cuspidal_coords.len(),
            "HeckeSummand: the ambient basis ({} vectors) and the cuspidal coordinates \
             ({} vectors) are two presentations of the same list of vectors and must \
             agree in length",
            ambient_basis.len(),
            cuspidal_coords.len()
        );
        Self {
            ambient_basis,
            cuspidal_coords,
        }
    }

    /// Dimension of the summand (inside sign-0 modular symbols, so twice
    /// the multiplicity-weighted degree of the eigensystem).
    pub fn dimension(&self) -> usize {
        self.cuspidal_coords.len()
    }

    /// Basis vectors in ambient quotient-basis coordinates.
    pub fn ambient_basis(&self) -> &[Vec<Rational>] {
        &self.ambient_basis
    }

    /// Basis vectors as coefficient vectors with respect to
    /// [`ModularSymbolsGamma0::cuspidal_basis`].
    pub fn cuspidal_coordinates(&self) -> &[Vec<Rational>] {
        &self.cuspidal_coords
    }
}

/// The full decomposition of the cuspidal subspace into Hecke-stable
/// summands, together with the primes that were used to split it.
#[derive(Debug, Clone)]
pub struct CuspidalHeckeDecomposition {
    level: u64,
    /// 0 for the full cuspidal space; +1 / -1 when the decomposition was
    /// computed on the corresponding star-involution eigenspace (each
    /// eigensystem then occurs ONCE instead of twice).
    sign: i8,
    summands: Vec<HeckeSummand>,
    /// The primes p (coprime to the level, in increasing order) whose T_p
    /// were used to refine the decomposition; the last three caused no
    /// refinement (stability criterion).
    split_primes: Vec<u64>,
}

impl CuspidalHeckeDecomposition {
    /// The level N.
    pub fn level(&self) -> u64 {
        self.level
    }

    /// The star-involution sign of the decomposed space: 0 for the full
    /// cuspidal space, +1 / -1 for a star eigenspace quotient.
    pub fn sign(&self) -> i8 {
        self.sign
    }

    /// The summands, ordered deterministically (by dimension, then by the
    /// characteristic polynomials of the splitting operators).
    pub fn summands(&self) -> &[HeckeSummand] {
        &self.summands
    }

    /// The primes used for the splitting (all coprime to the level).
    pub fn split_primes(&self) -> &[u64] {
        &self.split_primes
    }
}

/// Deterministic primality test by trial division (inputs are small).
pub(crate) fn is_prime_u64(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// The smallest prime q > p with gcd(q, n) = 1.
pub(crate) fn next_good_prime(p: u64, n: u64) -> u64 {
    let mut q = p + 1;
    loop {
        if is_prime_u64(q) && !n.is_multiple_of(q) {
            return q;
        }
        q += 1;
    }
}

/// Factor a monic polynomial over Q into monic irreducible factors with
/// multiplicities, SELF-CERTIFIED: denominators are cleared, the integer
/// polynomial is factored by Zassenhaus, and the claimed factorization is
/// verified by exact re-multiplication both over Z and (after making the
/// factors monic) over Q.  Quadratic factors additionally get an
/// unconditional irreducibility certificate (non-square discriminant).
pub(crate) fn factor_monic_rational_certified(
    f: &UnivariatePolynomial<Rational>,
) -> Result<Vec<(UnivariatePolynomial<Rational>, u32)>, String> {
    let deg = f
        .degree()
        .ok_or_else(|| "cannot factor the zero polynomial".to_string())?;
    if !f.is_monic() {
        return Err("factor_monic_rational_certified expects a monic polynomial".to_string());
    }
    if deg == 0 {
        return Ok(Vec::new());
    }
    // clear denominators: L = lcm of the coefficient denominators
    let mut l = Integer::one();
    for c in f.coefficients() {
        l = l.lcm(c.denominator());
    }
    let fz: Vec<Integer> = f
        .coefficients()
        .iter()
        .map(|c| &(c.numerator() * &l) / c.denominator())
        .collect();
    let (content, zfactors) = zassenhaus::factor(&fz)
        .map_err(|_| "Zassenhaus recombination exceeded its factor-count bound".to_string())?;
    // certificate over Z: content * prod g_i^{e_i} == L * f exactly
    let mut prod = vec![content.clone()];
    for (g, e) in &zfactors {
        for _ in 0..*e {
            prod = zx::mul(&prod, g);
        }
    }
    if zx::trim(&prod) != zx::trim(&fz) {
        return Err("factorization certificate FAILED: re-multiplied product differs over Z".to_string());
    }
    let degsum: usize = zfactors
        .iter()
        .map(|(g, e)| (g.len() - 1) * (*e as usize))
        .sum();
    if degsum != deg {
        return Err("factorization certificate FAILED: factor degrees do not sum".to_string());
    }
    let mut out: Vec<(UnivariatePolynomial<Rational>, u32)> = Vec::with_capacity(zfactors.len());
    for (g, e) in &zfactors {
        // unconditional irreducibility certificate for quadratics: an integer
        // quadratic has a rational root iff its discriminant is a perfect square
        if g.len() == 3 {
            let disc = &(&g[1] * &g[1]) - &(&(&Integer::from(4) * &g[2]) * &g[0]);
            if disc.signum() >= 0 && disc.is_perfect_square() {
                return Err(
                    "irreducibility certificate FAILED: quadratic factor has square discriminant"
                        .to_string(),
                );
            }
        }
        let lead = g.last().expect("factors are nonempty").clone();
        let coeffs: Vec<Rational> = g
            .iter()
            .map(|a| Rational::new(a.clone(), lead.clone()).expect("nonzero leading coefficient"))
            .collect();
        out.push((UnivariatePolynomial::new(coeffs), *e));
    }
    // certificate over Q: the monic factors re-multiply to the monic input
    let mut qprod = UnivariatePolynomial::new(vec![Rational::one()]);
    for (g, e) in &out {
        for _ in 0..*e {
            qprod = qprod * g.clone();
        }
    }
    if &qprod != f {
        return Err("factorization certificate FAILED: monic re-multiplication differs over Q".to_string());
    }
    Ok(out)
}

/// The matrix of a linear map restricted to the span of `basis` (column
/// vectors in the coordinates `t` acts on): solves C X = T C exactly.
/// Errors if the basis is linearly dependent or the span is not invariant.
pub(crate) fn restrict_to_column_span(
    t: &Matrix<Rational>,
    basis: &[Vec<Rational>],
) -> Result<Matrix<Rational>, String> {
    let k = basis.len();
    if k == 0 {
        return Ok(Matrix::zeros(0, 0));
    }
    let d = basis[0].len();
    if t.rows() != d || t.cols() != d {
        return Err("restrict_to_column_span: dimension mismatch".to_string());
    }
    // tc[j] = T * basis[j]
    let mut tc = vec![vec![Rational::zero(); d]; k];
    for (v, out) in basis.iter().zip(tc.iter_mut()) {
        for (i, o) in out.iter_mut().enumerate() {
            let mut sum = Rational::zero();
            for (kk, vk) in v.iter().enumerate() {
                if vk.is_zero() {
                    continue;
                }
                sum = &sum + &(t.get(i, kk).expect("entry in range") * vk);
            }
            *o = sum;
        }
    }
    // rref of [C | TC]: pivots exactly 0..k iff C has full column rank and
    // every T C column lies in the span of C.
    let mut flat = Vec::with_capacity(d * 2 * k);
    for i in 0..d {
        for v in basis {
            flat.push(v[i].clone());
        }
        for v in &tc {
            flat.push(v[i].clone());
        }
    }
    let aug = Matrix::from_vec(d, 2 * k, flat).expect("augmented matrix shape");
    let rref = aug
        .reduced_row_echelon_form()
        .expect("exact rref over Q cannot fail");
    if rref.pivots != (0..k).collect::<Vec<_>>() {
        return Err(
            "subspace is not invariant under the operator (or its basis is dependent)".to_string(),
        );
    }
    let mut xflat = Vec::with_capacity(k * k);
    for i in 0..k {
        for j in 0..k {
            xflat.push(rref.matrix.get(i, k + j).expect("entry in range").clone());
        }
    }
    Ok(Matrix::from_vec(k, k, xflat).expect("k x k restriction"))
}

/// f(T) for a square matrix T, by Horner's rule.
pub(crate) fn eval_poly_at_matrix(
    f: &UnivariatePolynomial<Rational>,
    t: &Matrix<Rational>,
) -> Matrix<Rational> {
    let k = t.rows();
    let mut acc: Matrix<Rational> = Matrix::zeros(k, k);
    for c in f.coefficients().iter().rev() {
        acc = (acc * t.clone()).expect("square matrices of equal size");
        if !c.is_zero() {
            acc = (acc + Matrix::identity(k).scalar_mul(c)).expect("same shape");
        }
    }
    acc
}

/// M^e by repeated multiplication (exponents here are tiny).
pub(crate) fn mat_pow(m: &Matrix<Rational>, e: u32) -> Matrix<Rational> {
    let mut acc: Matrix<Rational> = Matrix::identity(m.rows());
    for _ in 0..e {
        acc = (acc * m.clone()).expect("square matrices of equal size");
    }
    acc
}

/// sum_j x_j * basis[j] (coefficients x in the coordinates of `basis`).
pub(crate) fn lin_comb(basis: &[Vec<Rational>], x: &[Rational]) -> Vec<Rational> {
    let d = basis.first().map_or(0, Vec::len);
    let mut out = vec![Rational::zero(); d];
    for (c, v) in x.iter().zip(basis.iter()) {
        if c.is_zero() {
            continue;
        }
        for (o, w) in out.iter_mut().zip(v.iter()) {
            *o = &*o + &(c * w);
        }
    }
    out
}

impl ModularSymbolsGamma0 {
    /// Decompose the cuspidal subspace into Hecke-stable summands on which
    /// every examined T_p (p coprime to N) acts with a single irreducible
    /// characteristic factor.  See the module docs for the algorithm, the
    /// meaning of the multiplicities, and the honest limits (old/new
    /// splitting inside a shared eigensystem is deferred to degeneracy
    /// maps).
    ///
    /// The refinement uses successive primes coprime to N and stops after
    /// three consecutive primes cause no further splitting; the returned
    /// [`CuspidalHeckeDecomposition::split_primes`] records them.  Summands
    /// are ordered deterministically by dimension, then by the
    /// characteristic polynomial coefficient lists of the splitting
    /// operators.
    pub fn cuspidal_hecke_decomposition(&self) -> Result<CuspidalHeckeDecomposition, String> {
        let s = self.cuspidal_dimension();
        let identity_basis: Vec<Vec<Rational>> = (0..s)
            .map(|i| {
                let mut v = vec![Rational::zero(); s];
                v[i] = Rational::one();
                v
            })
            .collect();
        self.hecke_decomposition_of_subspace(identity_basis, 0)
    }

    /// Decompose the +1 or -1 eigenspace of the star involution inside the
    /// cuspidal subspace into Hecke-stable summands (same algorithm and
    /// ordering as [`Self::cuspidal_hecke_decomposition`]).  Because the
    /// star involution commutes with every T_n, each eigenspace is
    /// Hecke-stable and carries every eigensystem exactly ONCE: the
    /// summands here have HALF the dimension of the sign-0 summands, with
    /// identical Hecke actions (the classical efficiency halving).
    pub fn cuspidal_star_hecke_decomposition(
        &self,
        sign: i8,
    ) -> Result<CuspidalHeckeDecomposition, String> {
        if sign != 1 && sign != -1 {
            return Err("sign must be +1 or -1".to_string());
        }
        let start = self.cuspidal_star_eigenspace(sign)?;
        self.hecke_decomposition_of_subspace(start, sign)
    }

    /// Shared refinement loop: decompose the span of `start` (a basis in
    /// cuspidal coordinates of a Hecke-stable subspace) under the T_p with
    /// p coprime to N.
    fn hecke_decomposition_of_subspace(
        &self,
        start: Vec<Vec<Rational>>,
        sign: i8,
    ) -> Result<CuspidalHeckeDecomposition, String> {
        let s = start.len();
        let n = self.level();
        if s == 0 {
            return Ok(CuspidalHeckeDecomposition {
                level: n,
                sign,
                summands: Vec::new(),
                split_primes: Vec::new(),
            });
        }
        let mut summands: Vec<Vec<Vec<Rational>>> = vec![start];
        let mut cached: Vec<(u64, Matrix<Rational>)> = Vec::new();
        let mut stable_streak = 0usize;
        let mut p = 1u64;
        while stable_streak < 3 {
            p = next_good_prime(p, n);
            let t = self.hecke_matrix_cuspidal(p);
            let mut refined = false;
            let mut next: Vec<Vec<Vec<Rational>>> = Vec::new();
            for w in &summands {
                let tw = restrict_to_column_span(&t, w)?;
                let cp = charpoly_berkowitz(&tw).expect("square matrix");
                let factors = factor_monic_rational_certified(&cp)?;
                if factors.len() <= 1 {
                    next.push(w.clone());
                    continue;
                }
                refined = true;
                for (f, e) in &factors {
                    let fdeg = f.degree().expect("irreducible factors have degree >= 1");
                    // generalized eigenspace = ker f(T)^e (primary decomposition)
                    let ker = mat_pow(&eval_poly_at_matrix(f, &tw), *e)
                        .kernel()
                        .expect("exact kernel over Q cannot fail");
                    if ker.len() != fdeg * (*e as usize) {
                        return Err(format!(
                            "generalized T_{p}-eigenspace has dimension {} != deg * mult = {}",
                            ker.len(),
                            fdeg * (*e as usize)
                        ));
                    }
                    next.push(ker.iter().map(|x| lin_comb(w, x)).collect());
                }
            }
            let total: usize = next.iter().map(Vec::len).sum();
            if total != s {
                return Err(format!(
                    "summand dimensions sum to {total} != cuspidal dimension {s}"
                ));
            }
            summands = next;
            stable_streak = if refined { 0 } else { stable_streak + 1 };
            cached.push((p, t));
        }
        let split_primes: Vec<u64> = cached.iter().map(|(q, _)| *q).collect();
        // deterministic order: dimension, then the concatenated charpoly
        // coefficient lists of the splitting operators restricted to the summand
        let mut keyed: Vec<(usize, Vec<Rational>, Vec<Vec<Rational>>)> = Vec::new();
        for w in summands {
            let mut key: Vec<Rational> = Vec::new();
            for (_, t) in &cached {
                let tw = restrict_to_column_span(t, &w)?;
                let cp = charpoly_berkowitz(&tw).expect("square matrix");
                key.extend(cp.coefficients().iter().cloned());
            }
            keyed.push((w.len(), key, w));
        }
        keyed.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        let cusp_basis = self.cuspidal_basis();
        let summands: Vec<HeckeSummand> = keyed
            .into_iter()
            .map(|(_, _, w)| HeckeSummand {
                ambient_basis: w.iter().map(|x| lin_comb(cusp_basis, x)).collect(),
                cuspidal_coords: w,
            })
            .collect();
        Ok(CuspidalHeckeDecomposition {
            level: n,
            sign,
            summands,
            split_primes,
        })
    }

    /// The matrix of T_n restricted to a summand, in the summand's basis.
    /// Errors if T_n does not preserve the summand (mathematically
    /// impossible for summands of [`Self::cuspidal_hecke_decomposition`],
    /// since all T_n commute; kept as a hard consistency check).
    pub fn hecke_matrix_on_summand(
        &self,
        summand: &HeckeSummand,
        n: u64,
    ) -> Result<Matrix<Rational>, String> {
        let t = self.hecke_matrix_cuspidal(n);
        restrict_to_column_span(&t, summand.cuspidal_coordinates())
    }

    /// The exact characteristic polynomial of T_n on a summand (monic,
    /// coefficients in increasing degree order).
    pub fn hecke_charpoly_on_summand(
        &self,
        summand: &HeckeSummand,
        n: u64,
    ) -> Result<UnivariatePolynomial<Rational>, String> {
        Ok(charpoly_berkowitz(&self.hecke_matrix_on_summand(summand, n)?)
            .expect("restricted Hecke matrix is square"))
    }

    /// How T_n acts on a summand: a single (rational or honest algebraic)
    /// eigenvalue when charpoly(T_n | summand) is a power of one irreducible
    /// polynomial, or [`SummandHeckeAction::Mixed`] with the distinct
    /// irreducible factors otherwise (possible only for gcd(n, N) > 1 on
    /// old-type summands; see the module docs).
    pub fn hecke_action_on_summand(
        &self,
        summand: &HeckeSummand,
        n: u64,
    ) -> Result<SummandHeckeAction, String> {
        let cp = self.hecke_charpoly_on_summand(summand, n)?;
        let mut factors = factor_monic_rational_certified(&cp)?;
        match factors.len() {
            0 => Err("a zero-dimensional summand has no Hecke eigenvalue".to_string()),
            1 => {
                let (f, _) = factors.pop().expect("one factor");
                if f.degree() == Some(1) {
                    // monic x + c0: the eigenvalue is -c0
                    Ok(SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(
                        -f.coeff(0).clone(),
                    )))
                } else {
                    Ok(SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(
                        f,
                    )))
                }
            }
            _ => Ok(SummandHeckeAction::Mixed(factors)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rational from a small signed integer.
    fn rat(k: i64) -> Rational {
        Rational::from_integer(Integer::from(k))
    }

    /// Monic polynomial from ascending integer coefficients.
    fn poly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| rat(c)).collect())
    }

    /// (x - a)^m, monic.
    fn linear_power(a: i64, m: usize) -> UnivariatePolynomial<Rational> {
        let mut acc = poly(&[1]);
        for _ in 0..m {
            acc = acc * poly(&[-a, 1]);
        }
        acc
    }

    /// The rational eigenvalue of T_n on a summand, panicking otherwise.
    fn rational_eigenvalue(m: &ModularSymbolsGamma0, w: &HeckeSummand, n: u64) -> Rational {
        match m.hecke_action_on_summand(w, n).unwrap() {
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(a)) => a,
            other => panic!("expected a rational eigenvalue for T_{n}, got {other:?}"),
        }
    }

    /// Exact check that BOTH roots of the real quadratic x^2 + bx + c lie in
    /// [-2 sqrt(p), 2 sqrt(p)] (the Ramanujan-Petersson bound for both real
    /// embeddings), entirely in rational arithmetic:
    /// disc = b^2 - 4c >= 0 (real), f(+-2 sqrt p) >= 0, and vertex inside,
    /// i.e. 4p + c >= 0 with (4p + c)^2 >= 4 p b^2, and b^2 <= 16 p.
    fn quadratic_roots_within_ramanujan(b: &Rational, c: &Rational, p: u64) -> bool {
        let four_p = rat(4 * p as i64);
        let disc = &(b * b) - &(&rat(4) * c);
        if disc < Rational::zero() {
            return false;
        }
        let s = &four_p + c; // f(2 sqrt p) + f(-2 sqrt p) = 2(4p + c), split below
        if s < Rational::zero() {
            return false;
        }
        let s2 = &s * &s;
        let b2_4p = &(b * b) * &four_p;
        if s2 < b2_4p {
            return false;
        }
        // vertex -b/2 within [-2 sqrt p, 2 sqrt p]: b^2 <= 16p
        (b * b) <= rat(16 * p as i64)
    }

    #[test]
    fn test_factor_certified_basics() {
        // (x - 2)^2 (x^2 + x - 1): mixed linear power and irreducible quadratic
        let f = linear_power(2, 2) * poly(&[-1, 1, 1]);
        let factors = factor_monic_rational_certified(&f).unwrap();
        let mut got = factors.clone();
        got.sort_by_key(|(g, _)| (g.degree(), format!("{g:?}")));
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], (poly(&[-2, 1]), 2));
        assert_eq!(got[1], (poly(&[-1, 1, 1]), 1));
        // constant polynomial: no factors
        assert!(factor_monic_rational_certified(&poly(&[1]))
            .unwrap()
            .is_empty());
        // non-monic input is refused
        assert!(factor_monic_rational_certified(&poly(&[1, 2])).is_err());
        // rational (non-integral) coefficients are handled: (x - 1/2)(x + 1/2)
        let half = Rational::new(Integer::from(-1), Integer::from(4)).unwrap();
        let g = UnivariatePolynomial::new(vec![half, Rational::zero(), Rational::one()]);
        let gf = factor_monic_rational_certified(&g).unwrap();
        assert_eq!(gf.len(), 2, "x^2 - 1/4 splits into two rational factors");
    }

    /// GATE (level 37, genus 2): two RATIONAL newforms.  Both isogeny
    /// classes were confirmed independently in python before this test was
    /// written: 37a: y^2 + y = x^3 - x has c4 = 48, Delta = 37; 37b:
    /// y^2 + y = x^3 + x^2 - 23x - 50 has c4 = 1120, Delta = 37^3 = 50653;
    /// both semistable with Delta supported at 37 and c4 coprime to 37,
    /// hence conductor 37.  a_p by DIRECT POINT COUNTING over F_p
    /// (nonsingular count at p = 37):
    ///   37a: a_2..a_13 = -2, -3, -2, -1, -5, -2  and a_37 = -1 (37 - 38);
    ///   37b: a_2..a_13 =  0,  1,  0, -1,  3, -4  and a_37 =  1 (37 - 36).
    /// Note a_7 = -1 for BOTH curves, so T_7 alone cannot split the space:
    /// this exercises the cross-prime intersection.
    #[test]
    fn test_decomposition_level_37_two_rational_newforms() {
        let m = ModularSymbolsGamma0::new(37);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(m.cuspidal_dimension(), 4);
        assert_eq!(dec.summands().len(), 2, "two rational newforms at 37");
        for w in dec.summands() {
            assert_eq!(w.dimension(), 2, "rational newform summand is 2-dim (+/- pair)");
        }
        let primes = [2u64, 3, 5, 7, 11, 13];
        let curve_37a: Vec<Rational> = [-2i64, -3, -2, -1, -5, -2].iter().map(|&a| rat(a)).collect();
        let curve_37b: Vec<Rational> = [0i64, 1, 0, -1, 3, -4].iter().map(|&a| rat(a)).collect();
        let mut systems: Vec<Vec<Rational>> = Vec::new();
        for w in dec.summands() {
            let mut sys = Vec::new();
            for &p in &primes {
                let a = rational_eigenvalue(&m, w, p);
                // charpoly(T_p | summand) = (x - a_p)^2, and Ramanujan a_p^2 <= 4p
                let cp = m.hecke_charpoly_on_summand(w, p).unwrap();
                let am = a.numerator().to_i64();
                assert!(a.denominator().is_one(), "integral eigenvalue");
                assert_eq!(cp, linear_power(am, 2), "charpoly(T_{p}) on summand at 37");
                assert!(am * am <= 4 * p as i64, "Ramanujan bound at p = {p}");
                sys.push(a);
            }
            systems.push(sys);
        }
        systems.sort();
        let mut expected = vec![curve_37a.clone(), curve_37b.clone()];
        expected.sort();
        assert_eq!(systems, expected, "the two eigensystems are exactly 37a and 37b");
        // U_37 on each summand: a_37 = -1 for 37a, +1 for 37b (point-counted)
        for w in dec.summands() {
            let a2 = rational_eigenvalue(&m, w, 2);
            let a37 = rational_eigenvalue(&m, w, 37);
            if a2 == rat(-2) {
                assert_eq!(a37, rat(-1), "a_37(37a)");
            } else {
                assert_eq!(a2, rat(0), "the other summand is 37b");
                assert_eq!(a37, rat(1), "a_37(37b)");
            }
        }
    }

    /// GATE (level 22 = 2 * 11, genus 2): the cuspidal space is ENTIRELY old,
    /// two degeneracy images f(q), f(q^2) of 11a (dim S_2(22) = g = 2 and the
    /// old space from level 11 already has dimension 2, so the new space is
    /// zero).  The two images share ONE eigensystem away from {2, 11}, so the
    /// T_p-refinement over p coprime to 22 must NOT split them: the honest
    /// result is a single 4-dimensional summand with charpoly(T_p) =
    /// (x - a_p)^4, a_p the point-counted 11a values (a_3, a_5, a_7, a_13 =
    /// -1, 1, -2, 4 for y^2 + y = x^3 - x^2 - 10x - 20 with c4 = 496,
    /// Delta = -11^5).  Splitting the two copies apart requires the
    /// degeneracy maps between levels 11 and 22 - NOT implemented, honestly
    /// deferred.
    ///
    /// At the bad indices the degeneracy-matrix theory gives, on the ordered
    /// basis (f(q), f(q^2)) of the old space:
    ///   U_2 = [[a_2, 1], [-2, 0]], charpoly x^2 - a_2 x + 2 = x^2 + 2x + 2
    ///   (a_2 = -2 point-counted; disc -4 < 0 so irreducible), and
    ///   U_11 = a_11 * identity with a_11 = 1 = 11 - #E_ns(F_11) (split
    ///   multiplicative, point-counted).  On sign-0 modular symbols each
    ///   charpoly appears squared.
    #[test]
    fn test_decomposition_level_22_old_11a_does_not_split() {
        let m = ModularSymbolsGamma0::new(22);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(m.cuspidal_dimension(), 4);
        assert_eq!(
            dec.summands().len(),
            1,
            "the two degeneracy images of 11a share one eigensystem away from 2, 11"
        );
        let w = &dec.summands()[0];
        assert_eq!(w.dimension(), 4);
        for (p, a) in [(3u64, -1i64), (5, 1), (7, -2), (13, 4)] {
            assert_eq!(rational_eigenvalue(&m, w, p), rat(a), "a_{p}(11a) at level 22");
            assert_eq!(
                m.hecke_charpoly_on_summand(w, p).unwrap(),
                linear_power(a, 4),
                "charpoly(T_{p}) = (x - a_p)^4 on the old summand"
            );
        }
        // U_2: honest degree-2 eigenvalue x^2 + 2x + 2, charpoly its square
        assert_eq!(
            m.hecke_action_on_summand(w, 2).unwrap(),
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(poly(&[2, 2, 1])))
        );
        assert_eq!(
            m.hecke_charpoly_on_summand(w, 2).unwrap(),
            poly(&[2, 2, 1]) * poly(&[2, 2, 1])
        );
        // U_11 acts as the scalar a_11 = 1
        assert_eq!(rational_eigenvalue(&m, w, 11), rat(1));
        assert_eq!(
            m.hecke_charpoly_on_summand(w, 11).unwrap(),
            linear_power(1, 4)
        );
    }

    /// GATE (level 23, genus 2): a single Galois-conjugate pair of newforms
    /// with coefficient field Q(sqrt(5)) - no elliptic curve to point-count.
    /// Every asserted polynomial was derived through a SECOND, fully
    /// disjoint path in python before this test was finalized: the cusp
    /// form F = eta(q)^2 eta(q^23)^2 lies in S_2(Gamma0(23)) by the Ligozat
    /// criteria, {F, T_2 F} is a q-expansion basis (dim = g = 2), the T_p
    /// matrices on that basis follow from the elementary coefficient rule
    /// (T_p f)_m = a_{pm} + p a_{m/p} certified on 60 exact coefficients
    /// (far beyond the Sturm bound 4 for level 23), and sympy factored the
    /// resulting charpolys:
    ///   T_2: x^2 + x - 1 (disc 5, the classical 23a eigenvalue polynomial;
    ///        embeddings (-1 +- sqrt 5)/2 ~ 0.618, -1.618, within 2 sqrt 2),
    ///   T_3: x^2 - 5,  T_5: x^2 + 2x - 4,  T_7: x^2 - 2x - 4 (disc 20
    ///        each: same field Q(sqrt 5)),
    ///   T_13: (x - 3)^2 - the eigenvalue is RATIONAL at 13 even though
    ///        the coefficient field has degree 2 (a_13 = 3 for both
    ///        conjugates), so the honest representation is Rational(3).
    /// On sign-0 modular symbols each of these appears squared.
    #[test]
    fn test_decomposition_level_23_quadratic_coefficient_field() {
        let m = ModularSymbolsGamma0::new(23);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(m.cuspidal_dimension(), 4);
        assert_eq!(dec.summands().len(), 1, "one Galois orbit of newforms at 23");
        let w = &dec.summands()[0];
        assert_eq!(w.dimension(), 4, "2 * [Q(sqrt 5) : Q]");
        // T_2: the eigenvalue IS the irreducible quadratic x^2 + x - 1
        let f2 = poly(&[-1, 1, 1]);
        assert_eq!(
            m.hecke_action_on_summand(w, 2).unwrap(),
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(f2.clone()))
        );
        assert_eq!(
            m.hecke_charpoly_on_summand(w, 2).unwrap(),
            f2.clone() * f2.clone(),
            "charpoly(T_2 | cusp at 23) = (x^2 + x - 1)^2"
        );
        // also the charpoly on the FULL cuspidal space (the summand is everything)
        assert_eq!(m.hecke_charpoly_cuspidal(2), f2.clone() * f2.clone());
        // a_4 = a_2^2 - 2 = -1 - a_2 (since a_2^2 = 1 - a_2) has the SAME
        // minimal polynomial: substituting x = -1 - y into x^2 + x - 1
        // gives y^2 + y - 1 again.
        assert_eq!(
            m.hecke_action_on_summand(w, 4).unwrap(),
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(f2.clone()))
        );
        // the eta-derived quadratic eigenvalue polynomials at 3, 5, 7
        for (p, fp) in [
            (3u64, poly(&[-5, 0, 1])),
            (5, poly(&[-4, 2, 1])),
            (7, poly(&[-4, -2, 1])),
        ] {
            assert_eq!(
                m.hecke_action_on_summand(w, p).unwrap(),
                SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(fp.clone())),
                "eigenvalue polynomial of T_{p} at 23"
            );
            let (c, b) = (fp.coeff(0).clone(), fp.coeff(1).clone());
            // trace = -b and norm = c are rational integers, disc = 5 * square
            assert!(b.denominator().is_one() && c.denominator().is_one());
            let disc = &(&b * &b) - &(&rat(4) * &c);
            let disc_int = disc.numerator().clone();
            let (q5, r5) = (&disc_int / &Integer::from(5), &disc_int % &Integer::from(5));
            assert!(
                r5.is_zero() && q5.signum() > 0 && q5.is_perfect_square(),
                "disc(T_{p} eigenvalue poly) = {disc_int} should be 5 * square"
            );
            assert!(
                quadratic_roots_within_ramanujan(&b, &c, p),
                "Ramanujan-Petersson fails for T_{p} at 23: x^2 + {b} x + {c}"
            );
            // charpoly = (eigenvalue polynomial)^2
            assert_eq!(
                m.hecke_charpoly_on_summand(w, p).unwrap(),
                fp.clone() * fp.clone()
            );
        }
        // T_13 is scalar: a_13 = 3 for both Galois conjugates (Ramanujan:
        // 9 <= 52), charpoly (x - 3)^4
        assert_eq!(rational_eigenvalue(&m, w, 13), rat(3));
        assert_eq!(
            m.hecke_charpoly_on_summand(w, 13).unwrap(),
            linear_power(3, 4)
        );
    }

    /// Level 44 = 4 * 11, genus 4: the old part from 11a now has THREE
    /// degeneracy images f(q), f(q^2), f(q^4) (divisors of 44/11 = 4) that
    /// share one eigensystem away from {2, 11} - a single 6-dimensional
    /// summand - plus a 2-dimensional rational newform summand (44a).  On
    /// the ordered old basis the degeneracy-matrix theory gives
    ///   U_2 = [[a_2, 1, 0], [-2, 0, 1], [0, 0, 0]] per sign copy,
    /// charpoly x (x^2 - a_2 x + 2) = x (x^2 + 2x + 2) with a_2 = -2
    /// point-counted on 11a, so on sign-0 symbols U_2 acts with the two
    /// DISTINCT irreducible factors x and x^2 + 2x + 2, each squared:
    /// the honest Mixed case (splitting it needs degeneracy maps).
    #[test]
    fn test_decomposition_level_44_mixed_up_action_on_old_summand() {
        let m = ModularSymbolsGamma0::new(44);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(m.cuspidal_dimension(), 8, "2 g(X0(44)) = 8");
        let dims: Vec<usize> = dec.summands().iter().map(HeckeSummand::dimension).collect();
        assert_eq!(dims, vec![2, 6], "44a newform + 6-dim old-11a block");
        let w2 = &dec.summands()[0];
        let w6 = &dec.summands()[1];
        // the 6-dim summand carries the 11a eigensystem at good primes
        for (p, a) in [(3u64, -1i64), (5, 1), (7, -2), (13, 4)] {
            assert_eq!(rational_eigenvalue(&m, w6, p), rat(a), "a_{p}(11a) at level 44");
            assert_eq!(
                m.hecke_charpoly_on_summand(w6, p).unwrap(),
                linear_power(a, 6)
            );
        }
        // U_2 on the old summand: honestly Mixed, factors {x, x^2 + 2x + 2}
        // with multiplicity 2 each
        let mut mixed = match m.hecke_action_on_summand(w6, 2).unwrap() {
            SummandHeckeAction::Mixed(fs) => fs,
            other => panic!("expected Mixed U_2 action on the old summand, got {other:?}"),
        };
        mixed.sort_by_key(|(g, _)| g.degree());
        assert_eq!(
            mixed,
            vec![(poly(&[0, 1]), 2), (poly(&[2, 2, 1]), 2)],
            "U_2 factors on the 3-image old block"
        );
        // U_11 is the scalar a_11 = 1 there
        assert_eq!(rational_eigenvalue(&m, w6, 11), rat(1));
        // the 2-dim summand is a rational newform with 4 | 44, so a_2 = 0
        // (U_p kills newforms with p^2 | N) and rational Ramanujan-bounded
        // eigenvalues at good primes
        assert_eq!(rational_eigenvalue(&m, w2, 2), rat(0), "a_2(44a) = 0 since 4 | 44");
        for p in [3u64, 5, 7, 13] {
            let a = rational_eigenvalue(&m, w2, p);
            assert!(a.denominator().is_one());
            let ai = a.numerator().to_i64();
            assert!(ai * ai <= 4 * p as i64, "Ramanujan for 44a at p = {p}");
        }
    }

    /// Level 45 = 9 * 5, genus 3: a 2-dim rational newform summand (45a)
    /// plus the 4-dim old block of 15a (images t = 1, 3 of 45/15 = 3).
    /// Away from 45 the block carries the point-counted 15a eigensystem
    /// (a_2, a_7, a_11, a_13 = -1, 0, -4, -2).  At p = 3 the situation
    /// DIFFERS from the level-22/44 towers because 3 divides the newform
    /// level 15 already: with f = 15a and a_3 = -1 (multiplicative), the
    /// coefficient rule a_{3n} = a_3 a_n holds for ALL n, so on the ordered
    /// basis (f(q), f(q^3)) per sign copy
    ///   U_3 f = a_3 f,  U_3 f(q^3) = f(q)  =>  U_3 = [[a_3, 1], [0, 0]],
    /// charpoly x (x - a_3) = x (x + 1): the honest Mixed factors
    /// {x, x + 1}, each squared on sign-0 symbols.  U_5 = a_5 = 1 is scalar
    /// on the whole block (5 || 45 and 5 | 15: U_5 commutes with the
    /// raising maps here).
    #[test]
    fn test_decomposition_level_45_old_block_u3_mixed() {
        let m = ModularSymbolsGamma0::new(45);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(m.cuspidal_dimension(), 6, "2 g(X0(45)) = 6");
        let dims: Vec<usize> = dec.summands().iter().map(HeckeSummand::dimension).collect();
        assert_eq!(dims, vec![2, 4], "45a newform + 4-dim old-15a block");
        let w4 = &dec.summands()[1];
        for (p, a) in [(2u64, -1i64), (7, 0), (11, -4), (13, -2)] {
            assert_eq!(
                rational_eigenvalue(&m, w4, p),
                rat(a),
                "a_{p}(15a) on the old block at level 45"
            );
        }
        let mut mixed = match m.hecke_action_on_summand(w4, 3).unwrap() {
            SummandHeckeAction::Mixed(fs) => fs,
            other => panic!("expected Mixed U_3 on the 45-old block, got {other:?}"),
        };
        mixed.sort_by(|(f, _), (g, _)| f.coefficients().cmp(g.coefficients()));
        assert_eq!(
            mixed,
            vec![(poly(&[0, 1]), 2), (poly(&[1, 1]), 2)],
            "U_3 factors {{x, x + 1}} on the old-15a block"
        );
        assert_eq!(
            m.hecke_charpoly_on_summand(w4, 3).unwrap(),
            poly(&[0, 1]) * poly(&[0, 1]) * poly(&[1, 1]) * poly(&[1, 1])
        );
        assert_eq!(rational_eigenvalue(&m, w4, 5), rat(1), "U_5 = a_5(15a) = 1");
    }

    /// CONSISTENCY BATTERIES over every genus > 0 level up to 40 (plus the
    /// genus-0 ones, whose decomposition must be empty):
    ///   * summand dimensions sum to 2 g(X0(N)) = cuspidal dimension;
    ///   * every summand is T_q-stable for THREE further primes beyond the
    ///     ones used in the splitting (restriction solvable at all);
    ///   * charpoly(T_q | summand) is a power of a SINGLE irreducible
    ///     factor f with deg(f) * multiplicity = dim(summand);
    ///   * Ramanujan-Petersson at the good primes q: rational eigenvalues
    ///     satisfy a^2 <= 4q, quadratic ones pass the exact two-embedding
    ///     interval check, and any factor of degree d has
    ///     trace^2 <= d^2 * 4q (|sum of d roots| <= d * 2 sqrt(q)).
    fn battery(levels: &[u64]) {
        for &n in levels {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            let total: usize = dec.summands().iter().map(HeckeSummand::dimension).sum();
            assert_eq!(total, m.cuspidal_dimension(), "dims sum to 2g at level {n}");
            if m.cuspidal_dimension() == 0 {
                assert!(dec.summands().is_empty(), "genus 0 => empty decomposition");
                continue;
            }
            let mut q = *dec.split_primes().last().expect("at least one split prime");
            for _ in 0..3 {
                q = next_good_prime(q, n);
                for w in dec.summands() {
                    // stability under an operator NOT used in the splitting
                    let tw = m
                        .hecke_matrix_on_summand(w, q)
                        .unwrap_or_else(|e| panic!("T_{q} not stable on summand at level {n}: {e}"));
                    let cp = charpoly_berkowitz(&tw).expect("square matrix");
                    let factors = factor_monic_rational_certified(&cp).unwrap();
                    assert_eq!(
                        factors.len(),
                        1,
                        "charpoly(T_{q}) on a stabilized summand at level {n} must be a power of one irreducible"
                    );
                    let (f, e) = &factors[0];
                    let d = f.degree().expect("degree >= 1");
                    assert_eq!(d * (*e as usize), w.dimension(), "deg * mult = dim at level {n}");
                    // Ramanujan-Petersson wrongness detectors (q is coprime to n)
                    match d {
                        1 => {
                            let a = -f.coeff(0).clone();
                            assert!(a.denominator().is_one(), "integral a_q at level {n}");
                            let ai = a.numerator().to_i64();
                            assert!(
                                ai * ai <= 4 * q as i64,
                                "Ramanujan fails at level {n}, T_{q}: a = {ai}"
                            );
                        }
                        2 => {
                            assert!(
                                quadratic_roots_within_ramanujan(f.coeff(1), f.coeff(0), q),
                                "Ramanujan fails at level {n}, T_{q}: {f:?}"
                            );
                        }
                        _ => {
                            // |trace| = |sum of d roots| <= 2 d sqrt(q):
                            // trace^2 <= 4 d^2 q, exactly
                            let tr = -f.coeff(d - 1).clone();
                            let bound = rat(4 * (d * d) as i64 * q as i64);
                            assert!(
                                (&tr * &tr) <= bound,
                                "trace bound fails at level {n}, T_{q}: {f:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_consistency_battery_levels_up_to_25() {
        battery(&[11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
    }

    #[test]
    fn test_consistency_battery_levels_26_to_40() {
        battery(&[26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]);
    }

    #[test]
    fn test_genus_zero_levels_decompose_to_nothing() {
        for n in [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 25] {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            assert!(dec.summands().is_empty(), "S_2(Gamma0({n})) = 0");
            assert!(dec.split_primes().is_empty());
        }
    }

    #[test]
    fn test_genus_one_levels_single_2dim_summand() {
        // every genus-1 level: exactly one summand, 2-dimensional, and the
        // split primes end with three stable ones
        for n in [11u64, 14, 15, 17, 19, 20, 21, 24, 27, 32, 36] {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            assert_eq!(m.cuspidal_dimension(), 2, "genus 1 at level {n}");
            assert_eq!(dec.summands().len(), 1);
            assert_eq!(dec.summands()[0].dimension(), 2);
            assert!(dec.split_primes().len() >= 3);
        }
    }
}
