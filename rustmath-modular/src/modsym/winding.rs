//! # The winding element and exact L(f, 1) vanishing
//!
//! The winding element is the modular symbol e = {0, oo} in M_2(Gamma0(N)).
//! This module builds, for each summand V_f of the cuspidal Hecke
//! decomposition ([`super::decomposition`]), the Hecke-equivariant projection
//! pi_f : M_2(Gamma0(N)) -> V_f, and decides EXACTLY whether L(g, 1) = 0
//! for the newform orbit g attached to V_f.
//!
//! ## The theorem this module relies on (Manin, Birch)
//!
//! Let f be a newform of weight 2 for Gamma0(N) (or, for an old summand,
//! the underlying newform of lower level), V_f the corresponding summand of
//! the cuspidal subspace of sign-0 modular symbols over Q, and pi_f the
//! projection of M_2(Gamma0(N); Q) onto V_f along the sum of all other
//! primary Hecke components.  Then
//!
//! > pi_f(e) = 0  if and only if  L(g, 1) = 0 for EVERY Galois conjugate g
//! > in the orbit of f  (for a RATIONAL newform: iff L(f, 1) = 0).
//!
//! Derivation (the precise argument, following Manin, *Parabolic points and
//! zeta-functions of modular curves* (1972), Birch, and Cremona,
//! *Algorithms for Modular Elliptic Curves*, sections 2.8 and 2.11; Stein,
//! *Modular Forms: A Computational Approach*, section 3.10):
//!
//! 1. The Mellin transform of f at s = 1 is the period integral over e:
//!    with the integration pairing <f, {a, b}> = int_a^b f(z) dz one has
//!    <f, e> = int_0^{i oo} f(z) dz = i L(f, 1) / (2 pi), because
//!    (2 pi)^{-s} Gamma(s) L(f, s) = int_0^oo f(iy) y^{s-1} dy (absolutely
//!    convergent for a weight-2 cusp form at s = 1, no boundary terms).
//! 2. The pairing satisfies <f|T_n, x> = <f, T_n x> (the Hecke operators
//!    are self-adjoint for it), so for the projection pi_f, which is by
//!    construction a POLYNOMIAL in the T_p (built below from the certified
//!    charpoly factorizations), <g, pi_f(x)> = <g|pi_f, x> = <g, x> for
//!    every eigenform g in the orbit of f: indeed g|pi_f = c g with a
//!    scalar c, and pairing against V_f (on which pi_f = id, and which
//!    pairs perfectly with the orbit span by Eichler-Shimura) forces c = 1.
//! 3. Hence <g, pi_f(e)> = i L(g, 1)/(2 pi) and, e being a RATIONAL symbol,
//!    <g-bar, pi_f(e)> = conj(<g, pi_f(e)>).  The orbit span (holomorphic
//!    and antiholomorphic halves) pairs perfectly with V_f tensor C
//!    (Eichler-Shimura isotypic decomposition; distinct isotypic pieces
//!    pair to zero by self-adjointness, and the total cuspidal pairing is
//!    perfect), so pi_f(e) = 0 iff all these pairings vanish iff
//!    L(g, 1) = 0 for every conjugate g.  This vanishing statement is
//!    EXACT: it is decided by rational linear algebra with no analytic
//!    input, and is this crate's only source of CERTIFIED ZEROS of
//!    L-values.
//!
//! For an old summand (the block of all degeneracy images f(dz), d | N/M,
//! of a newform f of level M | N), step 1 applies to each image:
//! <f(d .), e> = i L(f, 1)/(2 pi d) (substitute w = dz), so the criterion
//! reads: the projection of e onto the old block vanishes iff L(g, 1) = 0
//! for every conjugate g of the underlying LOWER-LEVEL newform.  (Asserted
//! as a cross-level consistency battery in the tests: the answer at level
//! N must equal the answer at level M.)
//!
//! ## Construction of the projection
//!
//! For each prime p used to split the decomposition (all coprime to N),
//! let c_p = charpoly(T_p | M_2) with certified factorization
//! c_p = f_p^{e_p} * g_p, where f_p is the single irreducible factor with
//! charpoly(T_p | V_f) = f_p^{m_p} (m_p <= e_p since V_f is T_p-stable and
//! its charpoly divides the ambient one).  By the extended Euclidean
//! algorithm in Q[x] (gcd(f_p^{e_p}, g_p) = 1), write
//! u f_p^{e_p} + v g_p = 1; then pi_p = (v g_p)(T_p) is the primary
//! projection onto ker f_p(T_p)^{e_p} along ker g_p(T_p): v g_p is 1 mod
//! f_p^{e_p} and 0 mod g_p, and M_2 = ker f_p(T_p)^{e_p} (+) ker g_p(T_p)
//! by Cayley-Hamilton (c_p(T_p) = 0) and coprimality.  The pi_p are
//! polynomials in commuting operators, so they commute, and
//! pi_f = prod_p pi_p is an idempotent that fixes V_f (V_f lies in every
//! ker f_p(T_p)^{e_p}).
//!
//! Its image is EXACTLY V_f: the other cuspidal summands are killed at the
//! split primes that separate them (certified by the decomposition), and
//! the Eisenstein/boundary part is killed at any good prime because T_p
//! acts there with generalized eigenvalue p + 1 (weight-2 Eisenstein
//! series for Gamma0(N) have a_p = p + 1 at p coprime to N), which is
//! never a root of f_p: roots of f_p have absolute value <= 2 sqrt(p)
//! < p + 1 by Ramanujan-Petersson when deg f_p = 1, and a rational root
//! of an irreducible f_p of degree > 1 cannot exist.  NONE of this is
//! trusted: [`ModularSymbolsGamma0::summand_projection_matrix`] verifies
//! at runtime that pi_f^2 = pi_f, that pi_f fixes the summand basis
//! vector-by-vector, and that every column of pi_f lies in the summand
//! span; the tests additionally verify that pi_f commutes with further
//! Hecke operators and the star involution and kills the other summands.
//! If the split primes are ever insufficient, further good primes are
//! drawn, with an honest error if a fixed bound is exceeded.
//!
//! Corresponds to the winding-element machinery behind
//! `sage.modular.modsym` (Cremona's analytic-rank-0 test,
//! `ModularSymbolsAmbient.rational_period_mapping`) and the MAGMA handbook
//! chapter "Modular Symbols" (`WindingElement` and the `LRatio` machinery,
//! up to normalization).

use super::decomposition::{
    eval_poly_at_matrix, factor_monic_rational_certified, lin_comb, next_good_prime, HeckeSummand,
};
use super::gamma0::ModularSymbolsGamma0;
use crate::cusps::Cusp;
use rustmath_core::Ring;
use rustmath_matrix::{charpoly_berkowitz, Matrix};
use rustmath_polynomials::UnivariatePolynomial;
use rustmath_rationals::Rational;

type QPoly = UnivariatePolynomial<Rational>;

/// Extended Euclid in Q[x] for coprime nonzero inputs: returns (u, v) with
/// u a + v b = 1.  Every division step and the final Bezout identity are
/// re-verified by exact polynomial arithmetic (self-certification;
/// `div_rem` comes from a crate this module treats as untrusted).
fn poly_bezout_certified(a: &QPoly, b: &QPoly) -> Result<(QPoly, QPoly), String> {
    if a.is_zero() || b.is_zero() {
        return Err("poly_bezout_certified: inputs must be nonzero".to_string());
    }
    let mut r0 = a.clone();
    let mut r1 = b.clone();
    let mut s0 = QPoly::new(vec![Rational::one()]);
    let mut s1 = QPoly::zero();
    let mut t0 = QPoly::zero();
    let mut t1 = QPoly::new(vec![Rational::one()]);
    while !r1.is_zero() {
        let (q, r) = r0
            .div_rem(&r1)
            .map_err(|e| format!("polynomial division failed: {e:?}"))?;
        // certify the division: q * r1 + r == r0 exactly
        if q.clone() * r1.clone() + r.clone() != r0 {
            return Err("polynomial div_rem certificate FAILED".to_string());
        }
        let s2 = s0 - q.clone() * s1.clone();
        let t2 = t0 - q * t1.clone();
        r0 = std::mem::replace(&mut r1, r);
        s0 = std::mem::replace(&mut s1, s2);
        t0 = std::mem::replace(&mut t1, t2);
    }
    // r0 = gcd (a unit for coprime inputs); normalize u a + v b = 1
    if r0.degree() != Some(0) {
        return Err(format!(
            "poly_bezout_certified: inputs are not coprime (gcd degree {:?})",
            r0.degree()
        ));
    }
    let c = r0.coeff(0).clone();
    let cinv = Rational::new(c.denominator().clone(), c.numerator().clone())
        .map_err(|e| format!("gcd constant not invertible: {e:?}"))?;
    let u = s0.scalar_mul(&cinv);
    let v = t0.scalar_mul(&cinv);
    // certify the Bezout identity
    let ident = u.clone() * a.clone() + v.clone() * b.clone();
    if ident != QPoly::new(vec![Rational::one()]) {
        return Err("Bezout certificate FAILED: u a + v b != 1".to_string());
    }
    Ok((u, v))
}

/// f^e by repeated multiplication.
fn poly_pow(f: &QPoly, e: u32) -> QPoly {
    let mut acc = QPoly::new(vec![Rational::one()]);
    for _ in 0..e {
        acc = acc * f.clone();
    }
    acc
}

/// Matrix-vector product (matrix acting on coordinate column vectors).
fn mat_vec(m: &Matrix<Rational>, v: &[Rational]) -> Vec<Rational> {
    (0..m.rows())
        .map(|i| {
            let mut s = Rational::zero();
            for (j, x) in v.iter().enumerate() {
                if x.is_zero() {
                    continue;
                }
                s = &s + &(m.get(i, j).expect("entry in range") * x);
            }
            s
        })
        .collect()
}

/// True iff every column of `m` lies in the span of the given (independent)
/// column vectors: rref of [B | m] must have pivots only in the B block.
fn columns_in_span(m: &Matrix<Rational>, basis: &[Vec<Rational>]) -> bool {
    let k = basis.len();
    let d = m.rows();
    let mut flat = Vec::with_capacity(d * (k + m.cols()));
    for i in 0..d {
        for v in basis {
            flat.push(v[i].clone());
        }
        for j in 0..m.cols() {
            flat.push(m.get(i, j).expect("entry in range").clone());
        }
    }
    let aug = Matrix::from_vec(d, k + m.cols(), flat).expect("augmented matrix shape");
    let rref = aug
        .reduced_row_echelon_form()
        .expect("exact rref over Q cannot fail");
    rref.pivots.iter().all(|&p| p < k)
}

impl ModularSymbolsGamma0 {
    /// The winding element e = {0, oo}, in ambient quotient-basis
    /// coordinates.  Its boundary is [oo] - [0], so it is NOT cuspidal
    /// whenever the cusps 0 and oo are inequivalent; the Hecke-equivariant
    /// projections of this module map it into the cuspidal summands.
    pub fn winding_element(&self) -> Vec<Rational> {
        self.modular_symbol(&Cusp::zero(), &Cusp::infinity())
    }

    /// The Hecke-equivariant projection pi_f of the FULL space
    /// M_2(Gamma0(N)) onto the given summand of the sign-0 cuspidal Hecke
    /// decomposition, as a matrix on ambient quotient coordinates.
    ///
    /// pi_f is a polynomial in the Hecke operators T_p at primes p coprime
    /// to N (see the module docs for the construction and why the split
    /// primes of the decomposition suffice).  Before returning, the matrix
    /// is verified at runtime to (a) be idempotent, (b) fix every summand
    /// basis vector, and (c) have all columns inside the summand span; if
    /// (c) fails with the decomposition's split primes, further good primes
    /// are used, and an honest error is returned if 25 extra primes do not
    /// suffice (mathematically impossible per the module docs, kept as a
    /// hard wrongness detector).
    ///
    /// Errors on summands that do not come from the sign-0 decomposition of
    /// THIS space (e.g. star-quotient summands: their span is not a full
    /// primary component of M_2, so check (c) can never pass).
    pub fn summand_projection_matrix(
        &self,
        summand: &HeckeSummand,
    ) -> Result<Matrix<Rational>, String> {
        let dim = self.dimension();
        let k = summand.dimension();
        if k == 0 {
            return Err("cannot project onto a zero-dimensional summand".to_string());
        }
        if summand.ambient_basis().first().map(Vec::len) != Some(dim) {
            return Err("summand does not belong to this ambient space".to_string());
        }
        let mut pi: Matrix<Rational> = Matrix::identity(dim);
        let mut p = 1u64;
        let mut done = false;
        const EXTRA_PRIME_BOUND: usize = 25;
        for _ in 0..EXTRA_PRIME_BOUND {
            p = next_good_prime(p, self.level());
            let t = self.hecke_matrix(p);
            // irreducible factor of T_p on the summand (single, by the
            // decomposition's stability guarantee for good primes)
            let cp_summand = self.hecke_charpoly_on_summand(summand, p)?;
            let mut sfac = factor_monic_rational_certified(&cp_summand)?;
            if sfac.len() != 1 {
                return Err(format!(
                    "charpoly(T_{p} | summand) is not a power of one irreducible: \
                     not a summand of the Hecke decomposition"
                ));
            }
            let (f_p, _m_p) = sfac.pop().expect("one factor");
            // ambient charpoly and the multiplicity of f_p in it
            let c_full = charpoly_berkowitz(&t).expect("square matrix");
            let full_factors = factor_monic_rational_certified(&c_full)?;
            let e_p = full_factors
                .iter()
                .find(|(h, _)| *h == f_p)
                .map(|(_, e)| *e)
                .ok_or_else(|| {
                    format!(
                        "summand factor of T_{p} does not divide the ambient charpoly \
                         (impossible for an invariant subspace)"
                    )
                })?;
            // g_p = product of the other primary factors
            let mut g_p = QPoly::new(vec![Rational::one()]);
            for (h, e) in &full_factors {
                if *h != f_p {
                    g_p = g_p * poly_pow(h, *e);
                }
            }
            if g_p.degree() == Some(0) {
                // T_p has a single primary component: pi_p = identity,
                // nothing to multiply, but the prime still counts as used.
                continue;
            }
            let f_pow = poly_pow(&f_p, e_p);
            let (_u, v) = poly_bezout_certified(&f_pow, &g_p)?;
            let pi_p = eval_poly_at_matrix(&(v * g_p), &t);
            // runtime check: pi_p is idempotent
            let sq = (pi_p.clone() * pi_p.clone()).expect("square matrices of equal size");
            if sq != pi_p {
                return Err(format!("primary projection for T_{p} is not idempotent"));
            }
            pi = (pi * pi_p).expect("square matrices of equal size");
            // termination test: the accumulated image has shrunk to V_f
            // (the summand span); by construction it always CONTAINS V_f.
            if columns_in_span(&pi, summand.ambient_basis()) {
                done = true;
                break;
            }
        }
        if !done {
            return Err(format!(
                "projection image did not shrink to the summand within \
                 {EXTRA_PRIME_BOUND} good primes: is this a sign-0 summand of \
                 this space's cuspidal_hecke_decomposition()?"
            ));
        }
        // runtime check: idempotency of the product
        let sq = (pi.clone() * pi.clone()).expect("square matrices of equal size");
        if sq != pi {
            return Err("assembled projection is not idempotent".to_string());
        }
        // runtime check: pi fixes the summand basis vector-by-vector
        for v in summand.ambient_basis() {
            if mat_vec(&pi, v) != *v {
                return Err("assembled projection does not fix the summand".to_string());
            }
        }
        Ok(pi)
    }

    /// The projection pi_f(e) of the winding element e = {0, oo} onto the
    /// given summand, in ambient quotient-basis coordinates: EXACT rational
    /// coordinates, zero if and only if L(g, 1) = 0 for every Galois
    /// conjugate g of the newform attached to the summand (see the module
    /// docs for the theorem and its derivation).
    pub fn winding_projection(&self, summand: &HeckeSummand) -> Result<Vec<Rational>, String> {
        let pi = self.summand_projection_matrix(summand)?;
        Ok(mat_vec(&pi, &self.winding_element()))
    }

    /// EXACT vanishing of the L-value at s = 1: true iff L(g, 1) = 0 for
    /// every Galois conjugate g in the newform orbit of the summand (for a
    /// rational newform summand: iff L(f, 1) = 0; for an old summand: the
    /// criterion sees the underlying lower-level newform).
    ///
    /// This is the certified-zero source of the L-value lattice: a numeric
    /// computation can certify NONZERO but never zero; this exact statement
    /// can certify zero.
    pub fn l1_vanishes(&self, summand: &HeckeSummand) -> Result<bool, String> {
        Ok(self
            .winding_projection(summand)?
            .iter()
            .all(Rational::is_zero))
    }

    /// The exact rational the winding projection encodes on the plus
    /// quotient, for a summand whose star-plus part is ONE-dimensional
    /// (e.g. every rational-newform summand of the sign-0 decomposition).
    ///
    /// Returns (r, w) with pi_f(e) = r * w, where w (ambient coordinates)
    /// is the deterministic generator of the +1 star eigenspace inside the
    /// summand produced by the exact kernel routine on the summand basis.
    ///
    /// NORMALIZATION: r depends on the basis-dependent choice of w; it is
    /// NOT the Birch--Swinnerton-Dyer ratio L(f, 1)/Omega_f, which needs
    /// the period lattice (deliberately out of scope here).  What is
    /// basis-independent: whether r = 0 (iff [`Self::l1_vanishes`]), and
    /// ratios of r across elements written in the SAME w.  pi_f(e) always
    /// lies in the plus part since the star involution fixes e = {0, oo}
    /// (J fixes 0 and oo) and commutes with pi_f; this is re-verified at
    /// runtime.
    pub fn winding_ratio(
        &self,
        summand: &HeckeSummand,
    ) -> Result<(Rational, Vec<Rational>), String> {
        let pe = self.winding_projection(summand)?;
        // runtime check: the star involution fixes pi_f(e)
        let star = self.star_involution_matrix();
        if mat_vec(&star, &pe) != pe {
            return Err("star involution does not fix the winding projection".to_string());
        }
        // the +1 star eigenspace inside the summand, in summand coordinates
        let s_on_summand = {
            // solve star * B = B * X on the summand basis columns
            super::decomposition::restrict_to_column_span(&star, summand.ambient_basis())?
        };
        let k = summand.dimension();
        let id: Matrix<Rational> = Matrix::identity(k);
        let plus = (s_on_summand - id)
            .expect("same shape")
            .kernel()
            .expect("exact kernel over Q cannot fail");
        if plus.len() != 1 {
            return Err(format!(
                "winding_ratio needs a 1-dimensional plus part; this summand has \
                 dimension {} (orbit of degree > 1 or old block: use \
                 winding_projection directly)",
                plus.len()
            ));
        }
        let w_coords = &plus[0];
        let w_ambient = lin_comb(summand.ambient_basis(), w_coords);
        // express pi_f(e) = r * w exactly
        let mut r = Rational::zero();
        let mut seen = false;
        for (num, den) in pe.iter().zip(w_ambient.iter()) {
            if den.is_zero() {
                continue;
            }
            let cand = num / den;
            if !seen {
                r = cand;
                seen = true;
            } else if cand != r {
                return Err(
                    "winding projection is not proportional to the plus generator".to_string(),
                );
            }
        }
        if !seen {
            return Err(
                "plus generator is zero (impossible for a nonzero kernel vector)".to_string(),
            );
        }
        // exact certificate: pe == r * w (also catches pe = 0 => r must be 0)
        let scaled: Vec<Rational> = w_ambient.iter().map(|x| &r * x).collect();
        if scaled != pe {
            return Err("winding_ratio certificate FAILED: pi(e) != r * w".to_string());
        }
        Ok((r, w_ambient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modsym::decomposition::{HeckeEigenvalue, SummandHeckeAction};
    use crate::modsym::involutions::InvolutionAction;
    use rustmath_integers::Integer;

    fn rat(k: i64) -> Rational {
        Rational::from_integer(Integer::from(k))
    }

    /// a_2 of a summand when T_2 acts by a rational scalar, else None.
    fn rational_a2(m: &ModularSymbolsGamma0, w: &HeckeSummand) -> Option<Rational> {
        match m.hecke_action_on_summand(w, 2).unwrap() {
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(a)) => Some(a),
            _ => None,
        }
    }

    fn is_zero_vec(v: &[Rational]) -> bool {
        v.iter().all(Rational::is_zero)
    }

    #[test]
    fn test_poly_bezout_certified_basics() {
        let poly = |c: &[i64]| QPoly::new(c.iter().map(|&k| rat(k)).collect());
        // coprime: (x - 1) and (x + 1): u, v with u(x-1) + v(x+1) = 1
        let (u, v) = poly_bezout_certified(&poly(&[-1, 1]), &poly(&[1, 1])).unwrap();
        assert_eq!(
            u * poly(&[-1, 1]) + v * poly(&[1, 1]),
            poly(&[1]),
            "Bezout identity"
        );
        // non-coprime inputs are refused: (x - 1)(x + 1) and (x - 1)
        assert!(poly_bezout_certified(&poly(&[-1, 0, 1]), &poly(&[-1, 1])).is_err());
        // zero input refused
        assert!(poly_bezout_certified(&QPoly::zero(), &poly(&[1, 1])).is_err());
        // non-monic, rational-coefficient inputs are fine: 2x + 1 and 3x - 1
        let (u, v) = poly_bezout_certified(&poly(&[1, 2]), &poly(&[-1, 3])).unwrap();
        assert_eq!(u * poly(&[1, 2]) + v * poly(&[-1, 3]), poly(&[1]));
    }

    /// GATE (N = 11, rank 0): L(11a, 1) != 0, so the winding projection is
    /// NONZERO.  Independent derivation (python, before this test): a_p of
    /// y^2 + y = x^3 - x^2 - 10x - 20 (Delta = -11^5, c4 = 496, semistable
    /// => conductor 11) by direct point counting; the split-point-
    /// independence test pinned epsilon = +1 and the series gave
    /// L(11a, 1) = 0.2538418608559106843377589233509... (confirmed against
    /// a brute mpmath integral to 40 digits).
    #[test]
    fn test_winding_gate_level_11() {
        let m = ModularSymbolsGamma0::new(11);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(dec.summands().len(), 1);
        let w = &dec.summands()[0];
        let pe = m.winding_projection(w).unwrap();
        assert!(!is_zero_vec(&pe), "L(11a,1) != 0 => nonzero projection");
        assert!(!m.l1_vanishes(w).unwrap());
        // the projection lands in the cuspidal subspace even though the
        // winding element itself is NOT cuspidal at level 11
        assert!(!m.is_cuspidal(&m.winding_element()));
        assert!(m.is_cuspidal(&pe));
        // winding_ratio: nonzero rational, certificate pi(e) = r * w checked
        // internally
        let (r, w_gen) = m.winding_ratio(w).unwrap();
        assert!(!r.is_zero());
        assert!(!is_zero_vec(&w_gen));
    }

    /// GATE (N = 37): 37a (a_2 = -2, epsilon = -1, the first rank-1 curve)
    /// has L(37a, 1) = 0: the winding projection MUST vanish -- this is the
    /// chunk's certified zero.  37b (a_2 = 0, rank 0) has
    /// L(37b, 1) = 0.7256810619361527823362055410264... != 0 (python point
    /// counts + split-point test pinned epsilon(37a) = -1, epsilon(37b) =
    /// +1, values confirmed by brute mpmath integrals).
    #[test]
    fn test_winding_gate_level_37() {
        let m = ModularSymbolsGamma0::new(37);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(dec.summands().len(), 2);
        for w in dec.summands() {
            let a2 = rational_a2(&m, w).expect("rational newforms at 37");
            let vanish = m.l1_vanishes(w).unwrap();
            let (r, _) = m.winding_ratio(w).unwrap();
            if a2 == rat(-2) {
                assert!(vanish, "L(37a,1) = 0: winding projection must vanish");
                assert!(r.is_zero());
                // cross-check epsilon = -1 via the pinned Fricke sign
                assert_eq!(
                    m.atkin_lehner_on_summand(w, 37).unwrap(),
                    InvolutionAction::Scalar(1),
                    "w_37(37a) = +1 <=> epsilon = -1"
                );
            } else {
                assert_eq!(a2, rat(0), "the other summand is 37b");
                assert!(!vanish, "L(37b,1) != 0");
                assert!(!r.is_zero());
                assert_eq!(
                    m.atkin_lehner_on_summand(w, 37).unwrap(),
                    InvolutionAction::Scalar(-1),
                    "w_37(37b) = -1 <=> epsilon = +1"
                );
            }
        }
    }

    /// GATE (genus-1 rank-0 levels): every one of 14, 15, 17, 19, 20, 21,
    /// 24 has a nonvanishing L(f, 1), so a nonzero winding projection.
    /// Independent derivation (python, before this test): point counts on
    /// explicit models, epsilon pinned by split-point independence (+1 in
    /// every case), and the series values
    ///   L(14a,1) = 0.33022365934448053903...,
    ///   L(15a,1) = 0.35015076058315050580...,
    ///   L(17a,1) = 0.38676993838778004330...,
    ///   L(19a,1) = 0.45325324449610360358...,
    ///   L(20a,1) = 0.47072919032651896658...,
    ///   L(21a,1) = 0.45111540538849205559...,
    ///   L(24a,1) = 0.53912891187491080886...
    /// (each model certified against the level-N eigensystem: the
    /// functional-equation split-point test certifies (a_n, N, epsilon)
    /// jointly, and the point-counted a_p agree with this crate's
    /// independently gated eigenvalues; e.g. a_2(14a) = -1, a_7(14a) = 1,
    /// a_3(15a) = -1, a_5(15a) = 1 are pinned in the involutions tests).
    #[test]
    fn test_winding_gate_genus_one_rank_zero_levels() {
        for n in [14u64, 15, 17, 19, 20, 21, 24] {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            assert_eq!(dec.summands().len(), 1, "genus 1 at level {n}");
            let w = &dec.summands()[0];
            assert!(
                !m.l1_vanishes(w).unwrap(),
                "L(f,1) != 0 at rank-0 level {n}"
            );
            let (r, _) = m.winding_ratio(w).unwrap();
            assert!(!r.is_zero());
            // parity consistency: rank 0 forces epsilon = +1, i.e. w_N = -1
            assert_eq!(
                m.atkin_lehner_on_summand(w, n).unwrap(),
                InvolutionAction::Scalar(-1),
                "epsilon(level {n}) = +1"
            );
        }
    }

    /// GATE (N = 23, coefficient field Q(sqrt 5)): what parity forces.
    /// The Fricke sign is w_23 = -1 (asserted below from the exact
    /// matrix), so epsilon = -w_23 = +1 for BOTH embeddings: parity does
    /// NOT force vanishing, and the winding projection is the exact
    /// arbiter.  Independent numerics (python, eta product
    /// eta(z)^2 eta(23z)^2 = q^2 prod(1-q^n)^2 (1-q^23n)^2, T_2 charpoly
    /// x^2 + x - 1 certified on 100 coefficients, split-point test):
    ///   L(f_+, 1) = 0.55160578558263299341...  (a_2 = (-1+sqrt5)/2),
    ///   L(f_-, 1) = 0.45037937070981552574...  (a_2 = (-1-sqrt5)/2),
    /// both nonzero, so the projection must be NONZERO (it vanishes only
    /// if EVERY conjugate L-value vanishes).
    #[test]
    fn test_winding_gate_level_23_quadratic_orbit() {
        let m = ModularSymbolsGamma0::new(23);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(dec.summands().len(), 1);
        let w = &dec.summands()[0];
        assert_eq!(w.dimension(), 4);
        assert_eq!(
            m.atkin_lehner_on_summand(w, 23).unwrap(),
            InvolutionAction::Scalar(-1),
            "w_23 = -1 on the whole orbit => epsilon = +1: parity forces nothing"
        );
        assert!(!m.l1_vanishes(w).unwrap(), "both embeddings have L != 0");
        // winding_ratio needs a 1-dim plus part; the orbit has plus dim 2:
        // honest error
        assert!(m.winding_ratio(w).is_err());
    }

    /// CONSISTENCY (old blocks see the lower-level newform): the winding
    /// criterion at level N on the old block of a newform f of level M
    /// answers about L(f, 1) itself (module docs), so it must agree with
    /// the answer computed at level M.  All pairs here are rank-0, so both
    /// answers are `false`, but the EQUALITY is the theorem being tested.
    #[test]
    fn test_winding_old_blocks_match_lower_level() {
        for (n, m_level) in [
            (22u64, 11u64),
            (28, 14),
            (33, 11),
            (34, 17),
            (38, 19),
            (40, 20),
            (30, 15),
        ] {
            let mn = ModularSymbolsGamma0::new(n);
            let dec_n = mn.cuspidal_hecke_decomposition().unwrap();
            let old: Vec<&HeckeSummand> = dec_n
                .summands()
                .iter()
                .filter(|w| w.dimension() == 4)
                .collect();
            assert_eq!(old.len(), 1, "one 4-dim old block at level {n}");
            let vanish_old = mn.l1_vanishes(old[0]).unwrap();
            let mm = ModularSymbolsGamma0::new(m_level);
            let dec_m = mm.cuspidal_hecke_decomposition().unwrap();
            assert_eq!(dec_m.summands().len(), 1);
            let vanish_new = mm.l1_vanishes(&dec_m.summands()[0]).unwrap();
            assert_eq!(
                vanish_old, vanish_new,
                "old block at {n} vs newform at {m_level}"
            );
            assert!(!vanish_new, "L != 0 for the rank-0 newform at {m_level}");
        }
    }

    /// BATTERY over all levels: every summand's projection passes the full
    /// runtime verification (idempotent, fixes the summand, image inside
    /// the summand -- checked inside summand_projection_matrix), plus here:
    /// equivariance with three Hecke operators NOT used in the splitting
    /// and with the star involution, annihilation of every other summand,
    /// star-fixedness of the winding element, and the two exact
    /// expectations:
    ///   * Fricke = Scalar(+1) (epsilon = -1) IMPLIES l1_vanishes -- the
    ///     cross-validation of the chunk-5 Atkin-Lehner signs against the
    ///     winding machinery;
    ///   * l1_vanishes EXACTLY at the 37a summand and nowhere else in
    ///     11..40 (the first newform orbit with all-conjugate central
    ///     vanishing is 37a: every other rational newform here is rank 0
    ///     -- python-verified nonzero L for 11..24 and 37b above, epsilon
    ///     = +1 parity + nonzero projection for the quadratic orbits --
    ///     and old blocks inherit their rank-0 lower level).
    fn battery(levels: std::ops::RangeInclusive<u64>) {
        for n in levels {
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            if dec.summands().is_empty() {
                continue;
            }
            let e = m.winding_element();
            let star = m.star_involution_matrix();
            assert_eq!(mat_vec(&star, &e), e, "star fixes {{0, oo}} at {n}");
            // three good primes beyond the split set
            let mut extra = Vec::new();
            let mut q = *dec.split_primes().last().unwrap();
            for _ in 0..3 {
                q = super::next_good_prime(q, n);
                extra.push(q);
            }
            let projections: Vec<Matrix<Rational>> = dec
                .summands()
                .iter()
                .map(|w| m.summand_projection_matrix(w).unwrap())
                .collect();
            for (i, w) in dec.summands().iter().enumerate() {
                let pi = &projections[i];
                // equivariance beyond the construction primes
                for &qq in &extra {
                    let t = m.hecke_matrix(qq);
                    assert_eq!(
                        (pi.clone() * t.clone()).unwrap(),
                        (t * pi.clone()).unwrap(),
                        "pi T_{qq} != T_{qq} pi at level {n}"
                    );
                }
                assert_eq!(
                    (pi.clone() * star.clone()).unwrap(),
                    (star.clone() * pi.clone()).unwrap(),
                    "pi star != star pi at level {n}"
                );
                // pi kills every other summand
                for (j, v) in dec.summands().iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    for b in v.ambient_basis() {
                        assert!(
                            is_zero_vec(&mat_vec(pi, b)),
                            "pi_{i} does not kill summand {j} at level {n}"
                        );
                    }
                }
                let vanish = m.l1_vanishes(w).unwrap();
                // epsilon = -1 (Fricke +1) forces the exact zero
                if m.atkin_lehner_on_summand(w, n).unwrap() == InvolutionAction::Scalar(1) {
                    assert!(
                        vanish,
                        "epsilon = -1 summand at level {n} must have L(f,1) = 0"
                    );
                }
                // global expectation for 11..40
                let expected = n == 37 && rational_a2(&m, w) == Some(rat(-2));
                assert_eq!(
                    vanish,
                    expected,
                    "l1_vanishes at level {n}, summand {i} (dim {})",
                    w.dimension()
                );
                // rational newform summands: winding_ratio consistency
                if w.dimension() == 2 {
                    let (r, _) = m.winding_ratio(w).unwrap();
                    assert_eq!(r.is_zero(), vanish, "ratio zero iff vanishing at {n}");
                }
            }
        }
    }

    #[test]
    fn test_winding_battery_levels_11_to_25() {
        battery(11..=25);
    }

    #[test]
    fn test_winding_battery_levels_26_to_33() {
        battery(26..=33);
    }

    #[test]
    fn test_winding_battery_levels_34_to_40() {
        battery(34..=40);
    }

    /// Error paths: star-quotient summands are rejected (their span is not
    /// a full primary component of the ambient space), as are foreign
    /// summands.
    #[test]
    fn test_projection_rejects_non_sign0_summands() {
        let m = ModularSymbolsGamma0::new(11);
        let plus = m.cuspidal_star_hecke_decomposition(1).unwrap();
        assert_eq!(plus.summands().len(), 1);
        assert!(
            m.summand_projection_matrix(&plus.summands()[0]).is_err(),
            "plus-quotient summand must be rejected (image check cannot pass)"
        );
        // a summand of a DIFFERENT level's space: dimension mismatch
        let m14 = ModularSymbolsGamma0::new(14);
        let dec14 = m14.cuspidal_hecke_decomposition().unwrap();
        assert!(m.summand_projection_matrix(&dec14.summands()[0]).is_err());
    }
}
