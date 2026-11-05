//! # The star involution and the Atkin-Lehner involutions on modular symbols
//!
//! ## The star involution
//!
//! The map iota induced by {alpha, beta} -> {-alpha, -beta}, i.e. by the
//! matrix J = [[-1, 0], [0, 1]] acting on the upper half plane through
//! z -> -conj(z) (complex conjugation on X_0(N)).  On a weight-2 Manin
//! symbol [g] = {g(0), g(oo)} one has J(0) = 0 and J(oo) = oo, so
//!     iota [g] = {Jg(0), Jg(oo)} = {(JgJ)(0), (JgJ)(oo)} = [JgJ],
//! and for g = [[a, b], [c, d]] the conjugate JgJ = [[a, -b], [-c, d]] is in
//! SL2(Z) with bottom row (-c, d): on Manin symbols
//!     iota (c : d) = (-c : d),
//! with NO extra sign in weight 2 (for weight k the polynomial part picks up
//! P(X, Y) -> P(X, -Y); here P = 1).
//!
//! Well-definedness on the quotient by the Manin relations (derived, then
//! re-verified generator-by-generator in the tests): J Gamma0(N) J =
//! Gamma0(N), and conjugation by J sends S to S^{-1} = -S (the same point
//! map on P^1) and T to S T^2 S^{-1} modulo +-1, so the image of an
//! S-relation is an S-relation and the image of a T-relation is a
//! combination of S- and T-relations:
//!     iota(x(1 + T + T^2)) = y + (yS)T^2 S^{-1} + (yS)T S^{-1}
//!         = y - (yS)T^2 - (yS)T = y + yS = 0  (mod relations), y = iota x.
//!
//! Because complex conjugation is induced by an orientation-reversing
//! homeomorphism of the real surface X_0(N)(C) that fixes the (real) cusps,
//! its +1 and -1 eigenspaces on H_1(X_0(N), Q) both have dimension g (the
//! classical real-structure fact; Cremona, *Algorithms for Modular Elliptic
//! Curves*, section 2.1); the tests assert dim(cusp+) = dim(cusp-) = g.
//!
//! ## Atkin-Lehner involutions
//!
//! For an exact divisor Q || N (Q | N with gcd(Q, N/Q) = 1) the operator
//! W_Q is induced by any integer matrix
//!     W = [[Q a, b], [N c, Q d]]   with det W = Q,
//! i.e. Q a d - (N/Q) b c = 1 (solvable exactly because gcd(Q, N/Q) = 1),
//! acting on modular symbols by {alpha, beta} -> {W alpha, W beta}.
//!
//! * Well-definedness: W normalizes Gamma0(N) (direct computation), so the
//!   class of {W alpha, W beta} does not depend on the representative path.
//! * Choice-independence: two such matrices W, W' satisfy W^{-1} W' in
//!   Gamma0(N) (the quotient has integer entries, determinant 1 and lower
//!   left entry divisible by N), and Gamma0(N) acts trivially on the
//!   coinvariants, so the induced operator is the same; asserted on exact
//!   matrices in the tests.
//! * Involutivity: W^2 = Q * gamma with gamma in Gamma0(N), and scalar
//!   matrices act trivially on weight-2 symbols, so W_Q^2 = id on the whole
//!   space; asserted exactly.
//!
//! Sign conventions verified from scratch (see the test docs): the pairing
//! <f, {alpha, beta}> = int_alpha^beta f(z) dz satisfies
//! <f, W x> = <f|_2 W, x> (substitution rule, exact), so on the modular
//! symbol summand of a newform f the operator W_Q acts by the SAME scalar
//! w_Q as f|_2 W_Q = w_Q f.  The Fricke eigenvalues asserted at levels 11,
//! 14, 15 were derived exactly from the eta functional equation
//! eta(-1/z) = sqrt(-iz) eta(z) applied to the eta-product newforms (e.g.
//! f_11 = eta(z)^2 eta(11z)^2 gives f(-1/(11z)) = -11 z^2 f(z), so
//! w_11 = -1), re-verified numerically to 40 digits in python, with the eta
//! products themselves certified against direct point counts.
//!
//! Corresponds to `sage.modular.modsym.ambient` (`star_involution`,
//! `atkin_lehner_operator`) and the MAGMA handbook chapter "Modular
//! Symbols" (`AtkinLehnerOperator`, `StarInvolution`).

use super::decomposition::{restrict_to_column_span, HeckeSummand};
use super::gamma0::ModularSymbolsGamma0;
use crate::cusps::Cusp;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;

/// How an involution (star or Atkin-Lehner) acts on a subspace it preserves.
/// Since the operator squares to the identity, its restriction is
/// diagonalizable over Q with eigenvalues +-1 and is fully described by the
/// two eigenspace dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvolutionAction {
    /// The involution acts as the scalar +1 or -1 on the whole subspace.
    Scalar(i8),
    /// Both eigenvalues occur: the subspace splits into a plus-part and a
    /// minus-part of the given (both positive) dimensions.
    Split {
        plus_dimension: usize,
        minus_dimension: usize,
    },
}

/// The involution data of one Hecke summand: the star action and the action
/// of every Atkin-Lehner involution W_Q (Q > 1 an exact divisor of N).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummandInvolutions {
    /// Action of the star involution (always Split{d/2, d/2} on a summand of
    /// sign-0 modular symbols: each eigensystem occurs once per sign).
    pub star: InvolutionAction,
    /// (Q, action of W_Q) for every exact divisor Q > 1 of the level, in
    /// increasing order of Q (the last entry is the Fricke involution W_N).
    pub atkin_lehner: Vec<(u64, InvolutionAction)>,
}

/// gcd for u64.
fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd_u64(b, a % b)
    }
}

/// Extended gcd on i64: (g, x, y) with x*a + y*b = g = gcd(a, b) >= 0.
fn xgcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (1i64, 0i64);
    let (mut old_t, mut t) = (0i64, 1i64);
    while r != 0 {
        let q = old_r / r;
        (old_r, r) = (r, old_r - q * r);
        (old_s, s) = (s, old_s - q * s);
        (old_t, t) = (t, old_t - q * t);
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// Classify a matrix known to represent an involution on a subspace:
/// checks R^2 = I exactly, then splits into the +-1 eigenspace dimensions.
fn involution_action(r: &Matrix<Rational>) -> Result<InvolutionAction, String> {
    let k = r.rows();
    if r.cols() != k {
        return Err("involution matrix must be square".to_string());
    }
    if k == 0 {
        return Ok(InvolutionAction::Split {
            plus_dimension: 0,
            minus_dimension: 0,
        });
    }
    let id: Matrix<Rational> = Matrix::identity(k);
    let square = (r.clone() * r.clone()).expect("square matrices of equal size");
    if square != id {
        return Err("operator does not square to the identity on the subspace".to_string());
    }
    let plus = (r.clone() - id.clone())
        .expect("same shape")
        .kernel()
        .expect("exact kernel over Q cannot fail")
        .len();
    let minus = (r.clone() + id)
        .expect("same shape")
        .kernel()
        .expect("exact kernel over Q cannot fail")
        .len();
    if plus + minus != k {
        return Err(format!(
            "eigenspace dimensions {plus} + {minus} != {k} for an involution"
        ));
    }
    Ok(match (plus, minus) {
        (_, 0) => InvolutionAction::Scalar(1),
        (0, _) => InvolutionAction::Scalar(-1),
        _ => InvolutionAction::Split {
            plus_dimension: plus,
            minus_dimension: minus,
        },
    })
}

impl ModularSymbolsGamma0 {
    /// The matrix of the star involution iota: {alpha, beta} ->
    /// {-alpha, -beta} on the ambient quotient basis (acting on coordinate
    /// column vectors), via the Manin-symbol formula (c : d) -> (-c : d)
    /// derived in the module docs.
    pub fn star_involution_matrix(&self) -> Matrix<Rational> {
        let dim = self.dimension();
        let cols: Vec<&[Rational]> = self
            .basis_manin_indices()
            .iter()
            .map(|&j| {
                let (c, d) = self.manin_generator(j);
                let idx = self
                    .p1()
                    .index_of(-(c as i64), d as i64)
                    .expect("(-c : d) is a valid point of P^1");
                self.manin_generator_coords(idx)
            })
            .collect();
        let mut flat = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for col in &cols {
                flat.push(col[i].clone());
            }
        }
        Matrix::from_vec(dim, dim, flat).expect("dim x dim star matrix")
    }

    /// The star involution restricted to the cuspidal subspace, in the
    /// coordinates of [`Self::cuspidal_basis`].  (Star preserves the
    /// cuspidal subspace: the boundary of iota x is the image of the
    /// boundary of x under the cusp map [alpha] -> [-alpha].)
    pub fn star_matrix_cuspidal(&self) -> Result<Matrix<Rational>, String> {
        restrict_to_column_span(&self.star_involution_matrix(), self.cuspidal_basis())
    }

    /// Basis of the +1 (sign = 1) or -1 (sign = -1) eigenspace of the star
    /// involution on the FULL space M_2(Gamma0(N)), in ambient quotient
    /// coordinates.
    pub fn star_eigenspace_ambient(&self, sign: i8) -> Result<Vec<Vec<Rational>>, String> {
        if sign != 1 && sign != -1 {
            return Err("sign must be +1 or -1".to_string());
        }
        let s = self.star_involution_matrix();
        let id: Matrix<Rational> = Matrix::identity(self.dimension());
        let shifted = if sign == 1 {
            (s - id).expect("same shape")
        } else {
            (s + id).expect("same shape")
        };
        shifted
            .kernel()
            .map_err(|e| format!("kernel failed: {e:?}"))
    }

    /// Basis of the +1 / -1 eigenspace of the star involution on the
    /// CUSPIDAL subspace, as coefficient vectors with respect to
    /// [`Self::cuspidal_basis`].  Each has dimension g(X_0(N)) (see the
    /// module docs; asserted in the tests).
    pub fn cuspidal_star_eigenspace(&self, sign: i8) -> Result<Vec<Vec<Rational>>, String> {
        if sign != 1 && sign != -1 {
            return Err("sign must be +1 or -1".to_string());
        }
        let k = self.cuspidal_dimension();
        if k == 0 {
            return Ok(Vec::new());
        }
        let s = self.star_matrix_cuspidal()?;
        let id: Matrix<Rational> = Matrix::identity(k);
        let shifted = if sign == 1 {
            (s - id).expect("same shape")
        } else {
            (s + id).expect("same shape")
        };
        shifted
            .kernel()
            .map_err(|e| format!("kernel failed: {e:?}"))
    }

    /// The matrix of the sum of the given integral GL2(Q)+ matrices acting
    /// on modular symbols, as a map from `source` to `self` (columns indexed
    /// by the basis of `source`, rows by the basis of `self`): the j-th
    /// source basis element [g] = {g(0), g(oo)} maps to
    /// sum_M {M g(0), M g(oo)} converted to `self` coordinates via the
    /// Manin trick.  The CALLER is responsible for the sum being
    /// well-defined on the source quotient (verified generator-by-generator
    /// in the tests of every consumer: Atkin-Lehner and degeneracy maps).
    pub(crate) fn gl2_sum_action_from(
        &self,
        source: &ModularSymbolsGamma0,
        mats: &[[[Integer; 2]; 2]],
    ) -> Matrix<Rational> {
        let rows = self.dimension();
        let cols_n = source.dimension();
        let cols: Vec<Vec<Rational>> = source
            .basis_manin_indices()
            .iter()
            .map(|&j| self.gl2_sum_image_of_source_generator(source, mats, j))
            .collect();
        let mut flat = Vec::with_capacity(rows * cols_n);
        for i in 0..rows {
            for col in &cols {
                flat.push(col[i].clone());
            }
        }
        Matrix::from_vec(rows, cols_n, flat).expect("action matrix shape")
    }

    /// Image in `self` coordinates of the i-th Manin GENERATOR of `source`
    /// under sum_M {M g(0), M g(oo)} (used both to build the action matrix
    /// and, in tests, to verify well-definedness on non-basis generators).
    pub(crate) fn gl2_sum_image_of_source_generator(
        &self,
        source: &ModularSymbolsGamma0,
        mats: &[[[Integer; 2]; 2]],
        i: usize,
    ) -> Vec<Rational> {
        let g = source.p1().lift_to_sl2z(i);
        // [g] = {g(0), g(oo)} = {b/d, a/c}
        let alpha = Cusp::from_i64(g[0][1], g[1][1]);
        let beta = Cusp::from_i64(g[0][0], g[1][0]);
        let mut acc = vec![Rational::zero(); self.dimension()];
        for m in mats {
            let ma = alpha.apply_matrix(&m[0][0], &m[0][1], &m[1][0], &m[1][1]);
            let mb = beta.apply_matrix(&m[0][0], &m[0][1], &m[1][0], &m[1][1]);
            let img = self.modular_symbol(&ma, &mb);
            for (t, v) in acc.iter_mut().zip(img.iter()) {
                *t = &*t + v;
            }
        }
        acc
    }

    /// An integer matrix [[Q a, b], [N c, Q d]] with determinant Q for the
    /// Atkin-Lehner involution W_Q.  Errors unless Q || N (Q >= 1 divides N
    /// with gcd(Q, N/Q) = 1).
    pub(crate) fn atkin_lehner_integral_matrix(&self, q: u64) -> Result<[[Integer; 2]; 2], String> {
        let n = self.level();
        if q == 0 || !n.is_multiple_of(q) {
            return Err(format!("Q = {q} does not divide the level N = {n}"));
        }
        let m = n / q;
        if gcd_u64(q, m) != 1 {
            return Err(format!(
                "Q = {q} is not an exact divisor of N = {n}: gcd(Q, N/Q) = {} != 1",
                gcd_u64(q, m)
            ));
        }
        // q*x + (N/q)*y = 1; W = [[q x, y], [-N, q]] has det q(qx + (N/q)y) = q
        let (g, x, y) = xgcd_i64(q as i64, m as i64);
        debug_assert_eq!(g, 1);
        Ok([
            [Integer::from(q as i64 * x), Integer::from(y)],
            [Integer::from(-(n as i64)), Integer::from(q as i64)],
        ])
    }

    /// The matrix of the Atkin-Lehner involution W_Q on the ambient quotient
    /// basis (acting on coordinate column vectors).  Errors unless Q || N.
    /// See the module docs for well-definedness, choice-independence and
    /// involutivity (all re-asserted exactly in the tests).
    pub fn atkin_lehner_matrix(&self, q: u64) -> Result<Matrix<Rational>, String> {
        let w = self.atkin_lehner_integral_matrix(q)?;
        Ok(self.gl2_sum_action_from(self, &[w]))
    }

    /// The Atkin-Lehner involution W_Q restricted to the cuspidal subspace,
    /// in the coordinates of [`Self::cuspidal_basis`].
    pub fn atkin_lehner_matrix_cuspidal(&self, q: u64) -> Result<Matrix<Rational>, String> {
        restrict_to_column_span(&self.atkin_lehner_matrix(q)?, self.cuspidal_basis())
    }

    /// How the star involution acts on a summand of the cuspidal Hecke
    /// decomposition (star commutes with every T_p, so summands are
    /// star-stable; non-invariance is a hard error).  On sign-0 modular
    /// symbols this is always `Split{d/2, d/2}`.
    pub fn star_on_summand(&self, summand: &HeckeSummand) -> Result<InvolutionAction, String> {
        let s = self.star_matrix_cuspidal()?;
        let r = restrict_to_column_span(&s, summand.cuspidal_coordinates())?;
        involution_action(&r)
    }

    /// How the Atkin-Lehner involution W_Q acts on a summand of the
    /// cuspidal Hecke decomposition: `Scalar(+-1)` on a summand carrying a
    /// single newform orbit (e.g. the Fricke sign), or an honest
    /// `Split{...}` on old blocks where W_Q mixes the degeneracy images.
    pub fn atkin_lehner_on_summand(
        &self,
        summand: &HeckeSummand,
        q: u64,
    ) -> Result<InvolutionAction, String> {
        let w = self.atkin_lehner_matrix_cuspidal(q)?;
        let r = restrict_to_column_span(&w, summand.cuspidal_coordinates())?;
        involution_action(&r)
    }

    /// The full involution report of a summand: the star action and every
    /// Atkin-Lehner action W_Q for the exact divisors Q > 1 of the level.
    pub fn summand_involutions(
        &self,
        summand: &HeckeSummand,
    ) -> Result<SummandInvolutions, String> {
        let n = self.level();
        let star = self.star_on_summand(summand)?;
        let mut atkin_lehner = Vec::new();
        for q in 2..=n {
            if n.is_multiple_of(q) && gcd_u64(q, n / q) == 1 {
                atkin_lehner.push((q, self.atkin_lehner_on_summand(summand, q)?));
            }
        }
        Ok(SummandInvolutions { star, atkin_lehner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modsym::decomposition::{HeckeEigenvalue, SummandHeckeAction};
    use rustmath_matrix::charpoly_berkowitz;
    use rustmath_polynomials::UnivariatePolynomial;

    fn rat(k: i64) -> Rational {
        Rational::from_integer(Integer::from(k))
    }

    fn poly(coeffs: &[i64]) -> UnivariatePolynomial<Rational> {
        UnivariatePolynomial::new(coeffs.iter().map(|&c| rat(c)).collect())
    }

    fn mm(a: &Matrix<Rational>, b: &Matrix<Rational>) -> Matrix<Rational> {
        (a.clone() * b.clone()).expect("compatible shapes")
    }

    fn mv(a: &Matrix<Rational>, v: &[Rational]) -> Vec<Rational> {
        (0..a.rows())
            .map(|i| {
                let mut s = Rational::zero();
                for (j, x) in v.iter().enumerate() {
                    if x.is_zero() {
                        continue;
                    }
                    s = &s + &(a.get(i, j).expect("entry in range") * x);
                }
                s
            })
            .collect()
    }

    /// -alpha for a cusp (J alpha with J = [[-1, 0], [0, 1]]).
    fn neg_cusp(c: &Cusp) -> Cusp {
        match c {
            Cusp::Infinity => Cusp::Infinity,
            Cusp::Rational(p, q) => Cusp::new(-p.clone(), q.clone()),
        }
    }

    /// P^1 index of iota(i) = (-c : d) for the i-th generator.
    fn star_index(m: &ModularSymbolsGamma0, i: usize) -> usize {
        let (c, d) = m.manin_generator(i);
        m.p1()
            .index_of(-(c as i64), d as i64)
            .expect("(-c : d) is a valid P^1 point")
    }

    /// The rational eigenvalue of T_n on a summand, panicking otherwise.
    fn rational_eigenvalue(
        m: &ModularSymbolsGamma0,
        w: &crate::modsym::decomposition::HeckeSummand,
        n: u64,
    ) -> Rational {
        match m.hecke_action_on_summand(w, n).unwrap() {
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(a)) => a,
            other => panic!("expected a rational eigenvalue for T_{n}, got {other:?}"),
        }
    }

    /// GATE 1 for the star involution: the Manin-symbol formula
    /// (c : d) -> (-c : d) is WELL DEFINED on the quotient (the matrix
    /// reproduces the generator-level map on EVERY generator, not just the
    /// basis), squares to the identity, and agrees with the literature
    /// definition {alpha, beta} -> {-alpha, -beta} computed through the
    /// completely independent continued-fraction path (Manin trick).
    #[test]
    fn test_star_well_defined_involutive_matches_cusp_definition() {
        for n in [9u64, 11, 14, 15, 17, 19, 22, 23, 37, 44] {
            let m = ModularSymbolsGamma0::new(n);
            let s = m.star_involution_matrix();
            // well-definedness on every generator
            for i in 0..m.num_generators() {
                assert_eq!(
                    mv(&s, m.manin_generator_coords(i)),
                    m.manin_generator_coords(star_index(&m, i)).to_vec(),
                    "star not well defined at level {n}, generator {i}"
                );
            }
            // involutivity on the whole space M
            assert_eq!(
                mm(&s, &s),
                Matrix::identity(m.dimension()),
                "star^2 != id at level {n}"
            );
            // the cusp-path definition: iota {a, b} = {-a, -b}
            let pairs = [
                (Cusp::zero(), Cusp::infinity()),
                (Cusp::from_i64(1, 2), Cusp::from_i64(2, 3)),
                (Cusp::from_i64(-1, 3), Cusp::from_i64(3, 7)),
                (Cusp::from_i64(2, 5), Cusp::infinity()),
            ];
            for (a, b) in &pairs {
                assert_eq!(
                    mv(&s, &m.modular_symbol(a, b)),
                    m.modular_symbol(&neg_cusp(a), &neg_cusp(b)),
                    "star != {{-a, -b}} at level {n}"
                );
            }
            // the winding element {0, oo} is star-fixed (J fixes 0 and oo)
            let w = m.modular_symbol(&Cusp::zero(), &Cusp::infinity());
            assert_eq!(mv(&s, &w), w, "star must fix {{0, oo}} at level {n}");
        }
    }

    /// GATE 2: star commutes with every Hecke operator (including U_p at
    /// p | N), as exact ambient matrices.
    #[test]
    fn test_star_commutes_with_hecke() {
        for (n, ps) in [
            (11u64, vec![2u64, 3, 11]),
            (14, vec![2, 3, 7]),
            (15, vec![3, 5, 7]),
            (22, vec![2, 3, 11]),
            (23, vec![2, 5]),
            (37, vec![2, 5]),
        ] {
            let m = ModularSymbolsGamma0::new(n);
            let s = m.star_involution_matrix();
            for &p in &ps {
                let t = m.hecke_matrix(p);
                assert_eq!(
                    mm(&s, &t),
                    mm(&t, &s),
                    "star T_{p} != T_{p} star at level {n}"
                );
            }
        }
    }

    /// GATE 3: dim(cusp+) = dim(cusp-) = g for every level up to 45.
    /// Derivation: star is induced by complex conjugation on X_0(N)(C) (an
    /// orientation-reversing involution of a genus-g real surface whose
    /// real locus is nonempty - it contains the cusps), and on H_1 of such
    /// a surface the +1/-1 eigenspaces both have rank g.  The cuspidal
    /// dimension 2g itself was gated against an independent genus table in
    /// stage 1.  Also: the ambient +/- eigenspaces fill the whole space.
    #[test]
    fn test_star_eigenspace_dimensions_all_levels() {
        for n in 2..=45u64 {
            let m = ModularSymbolsGamma0::new(n);
            let g = m.cuspidal_dimension() / 2;
            let plus = m.cuspidal_star_eigenspace(1).unwrap();
            let minus = m.cuspidal_star_eigenspace(-1).unwrap();
            assert_eq!(plus.len(), g, "dim cusp+ != g at level {n}");
            assert_eq!(minus.len(), g, "dim cusp- != g at level {n}");
            let ap = m.star_eigenspace_ambient(1).unwrap();
            let am = m.star_eigenspace_ambient(-1).unwrap();
            assert_eq!(
                ap.len() + am.len(),
                m.dimension(),
                "M+ + M- != dim M at {n}"
            );
            if g > 0 {
                let sc = m.star_matrix_cuspidal().unwrap();
                assert_eq!(
                    mm(&sc, &sc),
                    Matrix::identity(2 * g),
                    "star^2 != id on cuspidal at {n}"
                );
            }
            // rejected sign
            assert!(m.cuspidal_star_eigenspace(0).is_err());
            assert!(m.cuspidal_star_eigenspace(2).is_err());
        }
    }

    /// GATE 4 (the efficiency halving): decomposing the +1 (or -1) star
    /// eigenspace alone reproduces the SAME Hecke eigensystems with all
    /// dimensions halved.  Expected eigenvalues are the independently
    /// derived stage-1 values (point counts for 11/22/37; the eta-product/
    /// sympy path for 23).
    #[test]
    fn test_plus_minus_quotients_reproduce_eigensystems() {
        // level 11: a_1..a_14 of 11a (point-counted primes, eta-verified
        // composites)
        let table11: [i64; 14] = [1, -2, -1, 2, 1, 2, -2, 0, -2, -2, 1, -2, 4, 4];
        let m11 = ModularSymbolsGamma0::new(11);
        for sign in [1i8, -1] {
            let dec = m11.cuspidal_star_hecke_decomposition(sign).unwrap();
            assert_eq!(dec.sign(), sign);
            assert_eq!(dec.summands().len(), 1);
            let w = &dec.summands()[0];
            assert_eq!(w.dimension(), 1, "each eigensystem appears ONCE per sign");
            for (i, &a) in table11.iter().enumerate() {
                assert_eq!(
                    rational_eigenvalue(&m11, w, (i + 1) as u64),
                    rat(a),
                    "a_{} of 11a on the sign-{sign} quotient",
                    i + 1
                );
            }
        }
        // level 22: the old 11a block halves to dimension 2; U_2 keeps the
        // honest algebraic eigenvalue x^2 + 2x + 2 (charpoly now the
        // quadratic itself, not its square)
        let m22 = ModularSymbolsGamma0::new(22);
        let dec = m22.cuspidal_star_hecke_decomposition(1).unwrap();
        assert_eq!(dec.summands().len(), 1);
        let w = &dec.summands()[0];
        assert_eq!(w.dimension(), 2);
        assert_eq!(
            m22.hecke_action_on_summand(w, 2).unwrap(),
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(poly(&[2, 2, 1])))
        );
        assert_eq!(
            m22.hecke_charpoly_on_summand(w, 2).unwrap(),
            poly(&[2, 2, 1])
        );
        for (p, a) in [(3u64, -1i64), (5, 1), (7, -2), (11, 1), (13, 4)] {
            assert_eq!(rational_eigenvalue(&m22, w, p), rat(a), "a_{p} at 22+");
        }
        // level 23: the Q(sqrt 5) newform orbit halves to dimension 2 with
        // the same irreducible eigenvalue polynomials
        let m23 = ModularSymbolsGamma0::new(23);
        let dec = m23.cuspidal_star_hecke_decomposition(1).unwrap();
        assert_eq!(dec.summands().len(), 1);
        let w = &dec.summands()[0];
        assert_eq!(w.dimension(), 2, "[Q(sqrt 5) : Q] = 2, once per sign");
        for (p, fp) in [
            (2u64, poly(&[-1, 1, 1])),
            (3, poly(&[-5, 0, 1])),
            (5, poly(&[-4, 2, 1])),
            (7, poly(&[-4, -2, 1])),
        ] {
            assert_eq!(
                m23.hecke_action_on_summand(w, p).unwrap(),
                SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Algebraic(fp.clone())),
                "T_{p} eigenvalue polynomial on the 23+ quotient"
            );
            assert_eq!(m23.hecke_charpoly_on_summand(w, p).unwrap(), fp);
        }
        assert_eq!(
            rational_eigenvalue(&m23, w, 13),
            rat(3),
            "rational a_13 = 3 at 23"
        );
        // level 37: both signs carry BOTH eigensystems (37a and 37b), each
        // 1-dimensional, matching the full decomposition eigenvalue by
        // eigenvalue
        let m37 = ModularSymbolsGamma0::new(37);
        let full = m37.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(full.sign(), 0);
        for sign in [1i8, -1] {
            let dec = m37.cuspidal_star_hecke_decomposition(sign).unwrap();
            assert_eq!(dec.summands().len(), 2);
            for w in dec.summands() {
                assert_eq!(w.dimension(), 1);
                let a2 = rational_eigenvalue(&m37, w, 2);
                // find the full summand with the same a_2 and compare
                let mate = full
                    .summands()
                    .iter()
                    .find(|v| rational_eigenvalue(&m37, v, 2) == a2)
                    .expect("matching full summand");
                assert_eq!(mate.dimension(), 2, "full summand is the +/- pair");
                for q in [3u64, 5, 7, 9, 37] {
                    assert_eq!(
                        m37.hecke_action_on_summand(w, q).unwrap(),
                        m37.hecke_action_on_summand(mate, q).unwrap(),
                        "sign-{sign} quotient reproduces T_{q} at 37"
                    );
                }
            }
        }
    }

    /// GATE 5: the Atkin-Lehner operator laws, all as exact matrix
    /// identities on the ambient space:
    ///   * W_Q is well defined (generator-by-generator re-verification);
    ///   * W_Q^2 = id;
    ///   * the operator does not depend on the choice of integral matrix
    ///     (every small solution of Q a d - (N/Q) b c = 1 gives the same
    ///     action);
    ///   * W_Q commutes with T_p for p coprime to N and with the star
    ///     involution;
    ///   * W_{Q1} W_{Q2} = W_{Q1 Q2} for coprime exact divisors;
    ///   * W_1 = id; invalid Q values give honest errors.
    #[test]
    fn test_atkin_lehner_involution_laws() {
        for (n, qs, good_ps) in [
            (11u64, vec![11u64], vec![2u64, 3]),
            (14, vec![2, 7, 14], vec![3, 5]),
            (15, vec![3, 5, 15], vec![2, 7]),
            (22, vec![2, 11, 22], vec![3, 5]),
            (37, vec![37], vec![2, 3]),
            (44, vec![4, 11, 44], vec![3, 5]),
            (45, vec![5, 9, 45], vec![2, 7]),
        ] {
            let m = ModularSymbolsGamma0::new(n);
            let id: Matrix<Rational> = Matrix::identity(m.dimension());
            let star = m.star_involution_matrix();
            for &q in &qs {
                let w = m.atkin_lehner_matrix(q).unwrap();
                // well-definedness on every generator
                let wm = m.atkin_lehner_integral_matrix(q).unwrap();
                for i in 0..m.num_generators() {
                    assert_eq!(
                        m.gl2_sum_image_of_source_generator(&m, std::slice::from_ref(&wm), i),
                        mv(&w, m.manin_generator_coords(i)),
                        "W_{q} not well defined at level {n}, generator {i}"
                    );
                }
                assert_eq!(mm(&w, &w), id, "W_{q}^2 != id at level {n}");
                for &p in &good_ps {
                    let t = m.hecke_matrix(p);
                    assert_eq!(mm(&w, &t), mm(&t, &w), "W_{q} T_{p} != T_{p} W_{q} at {n}");
                }
                assert_eq!(
                    mm(&w, &star),
                    mm(&star, &w),
                    "W_{q} star != star W_{q} at {n}"
                );
                // choice independence: brute-force alternative integral
                // matrices [[q a, b], [n c, q d]] with q a d - (n/q) b c = 1
                let (qi, ni) = (q as i64, n as i64);
                let mut found = 0;
                'search: for a in -4i64..=4 {
                    for b in -4i64..=4 {
                        for c in -4i64..=4 {
                            for d in -4i64..=4 {
                                if qi * a * d - (ni / qi) * b * c != 1 {
                                    continue;
                                }
                                let alt = [
                                    [Integer::from(qi * a), Integer::from(b)],
                                    [Integer::from(ni * c), Integer::from(qi * d)],
                                ];
                                let w_alt = m.gl2_sum_action_from(&m, &[alt]);
                                assert_eq!(
                                    w_alt, w,
                                    "different integral matrix for W_{q} at level {n} \
                                     gave a different operator"
                                );
                                found += 1;
                                if found >= 3 {
                                    break 'search;
                                }
                            }
                        }
                    }
                }
                assert!(
                    found >= 2,
                    "choice-independence needs >= 2 witnesses at ({n}, {q})"
                );
            }
            // product law for the coprime exact divisor pair (q1, q2)
            if qs.len() == 3 {
                let (q1, q2, q12) = (qs[0], qs[1], qs[2]);
                assert_eq!(q1 * q2, q12);
                let w1 = m.atkin_lehner_matrix(q1).unwrap();
                let w2 = m.atkin_lehner_matrix(q2).unwrap();
                let w12 = m.atkin_lehner_matrix(q12).unwrap();
                assert_eq!(mm(&w1, &w2), w12, "W_{q1} W_{q2} != W_{q12} at level {n}");
                assert_eq!(mm(&w2, &w1), w12, "Atkin-Lehner involutions commute at {n}");
            }
        }
        // W_1 = id; honest errors otherwise
        let m11 = ModularSymbolsGamma0::new(11);
        assert_eq!(
            m11.atkin_lehner_matrix(1).unwrap(),
            Matrix::identity(m11.dimension())
        );
        assert!(m11.atkin_lehner_matrix(7).is_err(), "7 does not divide 11");
        assert!(m11.atkin_lehner_matrix(0).is_err());
        let m44 = ModularSymbolsGamma0::new(44);
        assert!(
            m44.atkin_lehner_matrix(2).is_err(),
            "2 | 44 but gcd(2, 22) = 2: not an exact divisor"
        );
        let m45 = ModularSymbolsGamma0::new(45);
        assert!(m45.atkin_lehner_matrix(3).is_err(), "gcd(3, 15) = 3");
    }

    /// GATE 6: Atkin-Lehner SIGNS on rational newform summands.  Every
    /// asserted sign was derived independently BEFORE this test ran:
    ///
    ///   * Fricke w_N = -1 for 11a, 14a, 15a: from the eta functional
    ///     equation eta(-1/z) = sqrt(-iz) eta(z) applied to the eta
    ///     products eta(z)^2 eta(11z)^2, eta(z)eta(2z)eta(7z)eta(14z),
    ///     eta(z)eta(3z)eta(5z)eta(15z) - each gives
    ///     f(-1/(Nz)) = -N z^2 f(z) exactly, i.e. f|_2 W_N = -f (algebra
    ///     re-verified numerically to 40 digits; the eta products
    ///     themselves certified against direct point counts through p = 19).
    ///     The pairing argument in the module docs transports the SAME
    ///     scalar to the modular-symbol summand.
    ///   * prime exact divisors: the classical Atkin-Lehner relation
    ///     a_q = -w_q (weight 2, q || N, newform): with the point-counted
    ///     a_11(11a) = 1, a_2(14a) = -1, a_7(14a) = 1, a_3(15a) = -1,
    ///     a_5(15a) = 1 this forces w = -1, +1, -1, +1, -1 respectively,
    ///     and the products w_2 w_7 = w_14, w_3 w_5 = w_15 agree with the
    ///     eta-derived Fricke signs: three independent confirmations of
    ///     the sign convention.
    ///   * level 37: w_37 = -a_37 with the point-counted a_37(37a) = -1,
    ///     a_37(37b) = +1, so W_37 = +1 on the 37a summand (a_2 = -2) and
    ///     -1 on the 37b summand (a_2 = 0).  (Consistency remark: the
    ///     L-function sign epsilon = -w_37 is then -1 for 37a and +1 for
    ///     37b, matching ranks 1 and 0.)
    ///
    /// The relation w_q = -(U_q eigenvalue) is ALSO asserted directly with
    /// both sides computed in Rust.
    #[test]
    fn test_atkin_lehner_signs_on_rational_newforms() {
        // (level, expected AL report over exact divisors > 1)
        let expected: [(u64, Vec<(u64, i8)>); 3] = [
            (11, vec![(11, -1)]),
            (14, vec![(2, 1), (7, -1), (14, -1)]),
            (15, vec![(3, 1), (5, -1), (15, -1)]),
        ];
        for (n, als) in expected {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            assert_eq!(dec.summands().len(), 1);
            let w = &dec.summands()[0];
            let report = m.summand_involutions(w).unwrap();
            assert_eq!(
                report.star,
                InvolutionAction::Split {
                    plus_dimension: 1,
                    minus_dimension: 1
                },
                "star splits the 2-dim summand into +/- lines at {n}"
            );
            let expected_al: Vec<(u64, InvolutionAction)> = als
                .iter()
                .map(|&(q, s)| (q, InvolutionAction::Scalar(s)))
                .collect();
            assert_eq!(report.atkin_lehner, expected_al, "W_Q signs at level {n}");
            // w_q = -a_q at the prime exact divisors, both sides computed
            for &(q, s) in &als {
                if q < n {
                    let aq = rational_eigenvalue(&m, w, q);
                    assert_eq!(aq, rat(-(s as i64)), "a_{q} = -w_{q} at level {n}");
                }
            }
        }
        // level 11 Fricke also equals minus the U_11 eigenvalue
        let m11 = ModularSymbolsGamma0::new(11);
        let dec11 = m11.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(rational_eigenvalue(&m11, &dec11.summands()[0], 11), rat(1));
        // level 37: the two newforms get opposite Fricke signs -a_37
        let m37 = ModularSymbolsGamma0::new(37);
        let dec37 = m37.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(dec37.summands().len(), 2);
        for w in dec37.summands() {
            let a2 = rational_eigenvalue(&m37, w, 2);
            let a37 = rational_eigenvalue(&m37, w, 37);
            let sign = if a2 == rat(-2) { 1 } else { -1 }; // 37a : 37b
            assert_eq!(rat(-(sign as i64)), a37, "w_37 = -a_37 at level 37");
            assert_eq!(
                m37.atkin_lehner_on_summand(w, 37).unwrap(),
                InvolutionAction::Scalar(sign),
                "Fricke sign on the {} summand",
                if sign == 1 { "37a" } else { "37b" }
            );
        }
        // level 44 (4 || 44 composite exact divisor): every asserted number
        // was derived independently BEFORE this assertion was finalized.
        // The curve [0, 1, 0, 3, -1] (y^2 = x^3 + x^2 + 3x - 1) has
        // Delta = -2^8 * 11, c4 = -128: multiplicative at 11, additive at 2,
        // and its point counts match the (unique, dim-new = 2) rational
        // level-44 new eigensystem at every p <= 19, pinning the newform:
        //   a_3, a_5, a_7, a_13 = 1, -3, 2, -4 and a_11 = -1, a_2 = 0.
        // Anchors: w_11 = -a_11 = +1 (Atkin-Lehner, 11 || 44); w_44 = -1
        // from the functional-equation residual test
        // A(t) - w A(1/t) = (1 - w) A(1) on the point-counted coefficients
        // (pure numerics, no rank input); w_4 = w_44 / w_11 = -1.
        let m44 = ModularSymbolsGamma0::new(44);
        let dec44 = m44.cuspidal_hecke_decomposition().unwrap();
        let w2dim = &dec44.summands()[0];
        assert_eq!(w2dim.dimension(), 2);
        for (p, a) in [(3u64, 1i64), (5, -3), (7, 2), (13, -4)] {
            assert_eq!(
                rational_eigenvalue(&m44, w2dim, p),
                rat(a),
                "point-counted a_{p}(44a)"
            );
        }
        assert_eq!(
            rational_eigenvalue(&m44, w2dim, 11),
            rat(-1),
            "a_11(44a) = -1 = -w_11: the Atkin-Lehner anchor"
        );
        let report = m44.summand_involutions(w2dim).unwrap();
        assert_eq!(
            report.atkin_lehner,
            vec![
                (4, InvolutionAction::Scalar(-1)),
                (11, InvolutionAction::Scalar(1)),
                (44, InvolutionAction::Scalar(-1)),
            ],
            "44a Atkin-Lehner signs (product law (-1) * (+1) = -1 holds)"
        );
        // 45a: same programme.  The curve [1, -1, 0, 0, -5]
        // (y^2 + xy = x^3 - x^2 - 5) has Delta = -3^7 * 5, c4 = 9:
        // multiplicative at 5, additive at 3, point counts matching the
        // unique rational level-45 new eigensystem at every p <= 19:
        //   a_2, a_11, a_13 = 1, 4, -2 and a_5 = -1, a_3 = 0.
        // Anchors: w_5 = -a_5 = +1; w_45 = -1 by the same residual test;
        // w_9 = w_45 / w_5 = -1.
        let m45 = ModularSymbolsGamma0::new(45);
        let dec45 = m45.cuspidal_hecke_decomposition().unwrap();
        let w45a = &dec45.summands()[0];
        assert_eq!(w45a.dimension(), 2);
        for (p, a) in [(2u64, 1i64), (11, 4), (13, -2)] {
            assert_eq!(
                rational_eigenvalue(&m45, w45a, p),
                rat(a),
                "point-counted a_{p}(45a)"
            );
        }
        assert_eq!(
            rational_eigenvalue(&m45, w45a, 5),
            rat(-1),
            "a_5(45a) = -1 = -w_5: the Atkin-Lehner anchor"
        );
        assert_eq!(
            rational_eigenvalue(&m45, w45a, 3),
            rat(0),
            "a_3(45a) = 0 since 9 | 45"
        );
        let report = m45.summand_involutions(w45a).unwrap();
        assert_eq!(
            report.atkin_lehner,
            vec![
                (5, InvolutionAction::Scalar(1)),
                (9, InvolutionAction::Scalar(-1)),
                (45, InvolutionAction::Scalar(-1)),
            ],
            "45a Atkin-Lehner signs"
        );
    }

    /// GATE 7: Atkin-Lehner actions on OLD blocks are honestly non-scalar
    /// where theory says so.  On the level-22 old block of 11a (basis
    /// f(z), f(2z) per sign):
    ///   * W_11 acts through its level-11 eigenvalue on BOTH degeneracy
    ///     images: Scalar(-1) (the eta-derived w_11; cross-checked at the
    ///     matrix level by the intertwining test in the degeneracy module);
    ///   * W_2 SWAPS f(z) and (a multiple of) f(2z): eigenvalues +1 and -1
    ///     each with multiplicity 2, charpoly (x - 1)^2 (x + 1)^2;
    ///   * W_22 = W_2 W_11 = -W_2: the split mirrors.
    ///
    /// At level 45 the old block of 15a shows the same pattern with the
    /// LEVEL-15 eta-derived sign: W_5 = Scalar(-1) = w_5(15a), W_9 split.
    #[test]
    fn test_atkin_lehner_on_old_blocks() {
        let m22 = ModularSymbolsGamma0::new(22);
        let dec22 = m22.cuspidal_hecke_decomposition().unwrap();
        let old22 = &dec22.summands()[0];
        assert_eq!(old22.dimension(), 4);
        assert_eq!(
            m22.atkin_lehner_on_summand(old22, 11).unwrap(),
            InvolutionAction::Scalar(-1),
            "W_11 acts by the level-11 Fricke sign -1 on the whole old block"
        );
        assert_eq!(
            m22.atkin_lehner_on_summand(old22, 2).unwrap(),
            InvolutionAction::Split {
                plus_dimension: 2,
                minus_dimension: 2
            },
            "W_2 swaps the degeneracy images"
        );
        assert_eq!(
            m22.atkin_lehner_on_summand(old22, 22).unwrap(),
            InvolutionAction::Split {
                plus_dimension: 2,
                minus_dimension: 2
            }
        );
        // charpoly of W_2 on the old block is (x^2 - 1)^2
        let w2 = m22.atkin_lehner_matrix_cuspidal(2).unwrap();
        let r = restrict_to_column_span(&w2, old22.cuspidal_coordinates()).unwrap();
        assert_eq!(
            charpoly_berkowitz(&r).unwrap(),
            poly(&[-1, 0, 1]) * poly(&[-1, 0, 1]),
            "charpoly(W_2 | 22-old) = (x^2 - 1)^2"
        );
        // W_22 = W_2 W_11 = -W_2 on the old block (W_11 = -1 there)
        let w11 = m22.atkin_lehner_matrix_cuspidal(11).unwrap();
        let w22 = m22.atkin_lehner_matrix_cuspidal(22).unwrap();
        assert_eq!(mm(&w2, &w11), w22);
        // level 44 old block: W_11 = -1 again, W_4 and W_44 = -W_4 split
        let m44 = ModularSymbolsGamma0::new(44);
        let dec44 = m44.cuspidal_hecke_decomposition().unwrap();
        let old44 = &dec44.summands()[1];
        assert_eq!(old44.dimension(), 6);
        assert_eq!(
            m44.atkin_lehner_on_summand(old44, 11).unwrap(),
            InvolutionAction::Scalar(-1)
        );
        let w4_action = m44.atkin_lehner_on_summand(old44, 4).unwrap();
        let w44_action = m44.atkin_lehner_on_summand(old44, 44).unwrap();
        // W_44 = -W_4 on the block, so the split must mirror (recorded
        // values: W_4 = Split{4, 2}, W_44 = Split{2, 4})
        assert_eq!(
            w4_action,
            InvolutionAction::Split {
                plus_dimension: 4,
                minus_dimension: 2
            }
        );
        assert_eq!(
            w44_action,
            InvolutionAction::Split {
                plus_dimension: 2,
                minus_dimension: 4
            }
        );
        // level 45 old block of 15a: the level-15 sign w_5 = -1 transports
        let m45 = ModularSymbolsGamma0::new(45);
        let dec45 = m45.cuspidal_hecke_decomposition().unwrap();
        let old45 = &dec45.summands()[1];
        assert_eq!(old45.dimension(), 4);
        assert_eq!(
            m45.atkin_lehner_on_summand(old45, 5).unwrap(),
            InvolutionAction::Scalar(-1),
            "w_5(15a) = -1 acts on both degeneracy images at 45"
        );
        assert_eq!(
            m45.atkin_lehner_on_summand(old45, 9).unwrap(),
            InvolutionAction::Split {
                plus_dimension: 2,
                minus_dimension: 2
            }
        );
        // star splits every old block evenly too
        assert_eq!(
            m22.star_on_summand(old22).unwrap(),
            InvolutionAction::Split {
                plus_dimension: 2,
                minus_dimension: 2
            }
        );
        assert_eq!(
            m44.star_on_summand(old44).unwrap(),
            InvolutionAction::Split {
                plus_dimension: 3,
                minus_dimension: 3
            }
        );
    }
}

