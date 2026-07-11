//! # Degeneracy maps between levels and the old/new splitting
//!
//! For M | N and t | N/M there are two families of degeneracy maps between
//! the weight-2 modular symbol spaces:
//!
//! * LOWERING pi_t : M_2(Gamma0(N)) -> M_2(Gamma0(M)), induced directly by
//!   the matrix D_t = [[t, 0], [0, 1]]:  {alpha, beta} -> {t alpha, t beta}.
//!   Well-defined because D_t Gamma0(N) D_t^{-1} is contained in Gamma0(M):
//!   for gamma = [[a, b], [c, d]] with N | c the conjugate is
//!   [[a, t b], [c/t, d]], integral (t | N | c) with M | c/t (since tM | N).
//!
//! * RAISING delta_t : M_2(Gamma0(M)) -> M_2(Gamma0(N)), the group-homology
//!   TRANSFER composed with the action of T_t = [[1, 0], [0, t]].  Writing
//!   modular symbols as coinvariants V_Gamma (V = Div^0(P^1(Q))), the
//!   transfer V_Gamma -> V_Gamma' for a finite-index subgroup Gamma' is
//!   v -> sum_i g_i v where Gamma = union of the RIGHT cosets Gamma' g_i
//!   (well-defined: right multiplication permutes right cosets).  Here
//!   Gamma' = Gamma0(N/t) intersect Gamma^0(t) = T_t^{-1} Gamma0(N) T_t
//!   intersect SL2(Z), with EQUALITY T_t Gamma' T_t^{-1} = Gamma0(N) (both
//!   inclusions are immediate on entries, using tM | N), so
//!   delta_t(x) = sum_i (T_t g_i) x
//!   lands well-definedly in M_2(Gamma0(N)).  For t = 1 this is the plain
//!   transfer along Gamma0(N) < Gamma0(M).  Since conjugation by a fixed
//!   element preserves the Haar measure of SL2(R), Gamma' has the same
//!   index psi(N) in SL2(Z) as Gamma0(N), so the number of cosets is
//!   psi(N)/psi(M); the enumeration below certifies this count exactly.
//!
//! Exact composition laws verified in the tests (they certify the whole
//! construction):
//!     pi_1 delta_1 = [Gamma0(M) : Gamma0(N)] * id      (transfer degree)
//!     pi_t delta_t = psi(N)/psi(M) * id                (T_t D_t = t * I
//!                                                       acts trivially)
//!     pi_1 delta_t = pi_t delta_1 = T_t at level M     (t prime)
//! together with Hecke equivariance T_p pi/delta = pi/delta T_p for p
//! coprime to N, injectivity on the cuspidal subspace, and the span/block
//! structure of the old subspace (e.g. U_2 on the level-22 old space
//! satisfies U^2 + 2U + 2 = 0, explaining the stage-1 "Mixed"/algebraic
//! eigenvalues entirely through level-11 point counts).
//!
//! The NEW cuspidal subspace is the intersection of the kernels of all
//! lowering maps pi_1, pi_p to the levels N/p (p prime, p | N) - Stein,
//! *Modular Forms: A Computational Approach*, section 8.6; the old subspace
//! is the span of the images of the raising maps.  Both are computed here
//! exactly over Q.
//!
//! Coset representatives for Gamma'\Gamma0(M) are found by breadth-first
//! search using SCHREIER generators of Gamma0(M): with the coset transversal
//! r_i = lift_to_sl2z(i) of Gamma0(M)\SL2(Z) indexed by P^1(Z/M) and the
//! generators S, U = [[1,1],[0,1]] of SL2(Z), the elements r_i h r_j^{-1}
//! (j the coset of r_i h) generate Gamma0(M).  The BFS is certified: the
//! representative count must equal psi(N)/psi(M) and pairwise
//! inequivalence is re-checked explicitly.
//!
//! Corresponds to `sage.modular.modsym.ambient` (`degeneracy_map`) and the
//! MAGMA handbook chapter "Modular Symbols" (`DegeneracyMap`,
//! `NewSubspace`).

use super::decomposition::HeckeSummand;
use super::gamma0::ModularSymbolsGamma0;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;

/// 2x2 integer matrix, row major.
pub(crate) type IMat = [[Integer; 2]; 2];

/// Build an integer matrix from i64 entries.
pub(crate) fn imat(a: i64, b: i64, c: i64, d: i64) -> IMat {
    [
        [Integer::from(a), Integer::from(b)],
        [Integer::from(c), Integer::from(d)],
    ]
}

/// Product of two integer matrices.
pub(crate) fn imat_mul(x: &IMat, y: &IMat) -> IMat {
    [
        [
            &(&x[0][0] * &y[0][0]) + &(&x[0][1] * &y[1][0]),
            &(&x[0][0] * &y[0][1]) + &(&x[0][1] * &y[1][1]),
        ],
        [
            &(&x[1][0] * &y[0][0]) + &(&x[1][1] * &y[1][0]),
            &(&x[1][0] * &y[0][1]) + &(&x[1][1] * &y[1][1]),
        ],
    ]
}

/// Determinant.
fn imat_det(x: &IMat) -> Integer {
    &(&x[0][0] * &x[1][1]) - &(&x[0][1] * &x[1][0])
}

/// Inverse of a determinant-1 integer matrix (the adjugate).
fn imat_inv_det1(x: &IMat) -> IMat {
    debug_assert!(imat_det(x).is_one());
    [
        [x[1][1].clone(), -x[0][1].clone()],
        [-x[1][0].clone(), x[0][0].clone()],
    ]
}

/// psi(n) = n prod_{p | n} (1 + 1/p) = [SL2(Z) : Gamma0(n)].
fn psi_u64(n: u64) -> u64 {
    let mut result = n;
    let mut m = n;
    let mut p = 2;
    while p * p <= m {
        if m.is_multiple_of(p) {
            result += result / p;
            while m.is_multiple_of(p) {
                m /= p;
            }
        }
        p += 1;
    }
    if m > 1 {
        result += result / m;
    }
    result
}

/// The prime divisors of n, increasing.
fn prime_divisors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = 2;
    while p * p <= n {
        if n.is_multiple_of(p) {
            out.push(p);
            while n.is_multiple_of(p) {
                n /= p;
            }
        }
        p += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Schreier generators of Gamma0(M) from the P^1(Z/M) coset transversal of
/// Gamma0(M)\SL2(Z) and the generators S, U of SL2(Z).  Every returned
/// matrix is checked to lie in Gamma0(M).
fn gamma0_schreier_generators(space: &ModularSymbolsGamma0) -> Vec<IMat> {
    let p1 = space.p1();
    let m = space.level() as i64;
    let gens: [[[i64; 2]; 2]; 2] = [[[0, -1], [1, 0]], [[1, 1], [0, 1]]];
    let mut out = Vec::new();
    for i in 0..p1.len() {
        let ri64 = p1.lift_to_sl2z(i);
        let ri = imat(ri64[0][0], ri64[0][1], ri64[1][0], ri64[1][1]);
        for h in gens {
            let j = p1
                .apply_right(i, h)
                .expect("determinant-1 matrices act on P^1");
            let rj64 = p1.lift_to_sl2z(j);
            let rj = imat(rj64[0][0], rj64[0][1], rj64[1][0], rj64[1][1]);
            let hh = imat(h[0][0], h[0][1], h[1][0], h[1][1]);
            let cand = imat_mul(&imat_mul(&ri, &hh), &imat_inv_det1(&rj));
            assert!(
                imat_det(&cand).is_one() && (&cand[1][0] % &Integer::from(m)).is_zero(),
                "Schreier element must lie in Gamma0({m})"
            );
            out.push(cand);
        }
    }
    out
}

/// Right-coset representatives of Gamma' = Gamma0(n_big/t) intersect
/// Gamma^0(t) inside Gamma0(M) (M = `lower.level()`), by certified BFS:
/// Gamma0(M) = disjoint union of Gamma' g_i with exactly psi(N)/psi(M)
/// cosets, re-verified pairwise.
fn gamma_prime_coset_reps(
    lower: &ModularSymbolsGamma0,
    n_big: u64,
    t: u64,
) -> Result<Vec<IMat>, String> {
    let m = lower.level();
    let c_mod = Integer::from((n_big / t) as i64);
    let b_mod = Integer::from(t as i64);
    let in_gamma_prime = |x: &IMat| -> bool {
        imat_det(x).is_one() && (&x[1][0] % &c_mod).is_zero() && (&x[0][1] % &b_mod).is_zero()
    };
    let target = (psi_u64(n_big) / psi_u64(m)) as usize;
    let gens = gamma0_schreier_generators(lower);
    let mut reps: Vec<IMat> = vec![imat(1, 0, 0, 1)];
    let mut frontier: Vec<IMat> = vec![imat(1, 0, 0, 1)];
    while !frontier.is_empty() && reps.len() < target {
        let mut next = Vec::new();
        for x in &frontier {
            for g in &gens {
                let y = imat_mul(x, g);
                let known = reps
                    .iter()
                    .any(|r| in_gamma_prime(&imat_mul(&y, &imat_inv_det1(r))));
                if !known {
                    reps.push(y.clone());
                    next.push(y);
                    if reps.len() == target {
                        break;
                    }
                }
            }
        }
        frontier = next;
    }
    if reps.len() != target {
        return Err(format!(
            "coset BFS found {} representatives, expected psi({n_big})/psi({m}) = {target}",
            reps.len()
        ));
    }
    // certificate: pairwise inequivalent (the BFS already guarantees it,
    // but the check is cheap and unconditional)
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            if in_gamma_prime(&imat_mul(&reps[i], &imat_inv_det1(&reps[j]))) {
                return Err("coset representatives are not pairwise inequivalent".to_string());
            }
        }
    }
    Ok(reps)
}

impl ModularSymbolsGamma0 {
    /// The degeneracy LOWERING map pi_t : self -> lower for
    /// lower.level() | self.level() and t | (N/M):  {alpha, beta} ->
    /// {t alpha, t beta}.  Returns the matrix with columns indexed by the
    /// basis of `self` and rows by the basis of `lower`.
    pub fn degeneracy_lowering_matrix(
        &self,
        lower: &ModularSymbolsGamma0,
        t: u64,
    ) -> Result<Matrix<Rational>, String> {
        let (n, m) = (self.level(), lower.level());
        if m == 0 || !n.is_multiple_of(m) {
            return Err(format!("{m} does not divide the level {n}"));
        }
        if t == 0 || !(n / m).is_multiple_of(t) {
            return Err(format!("t = {t} does not divide N/M = {}", n / m));
        }
        Ok(lower.gl2_sum_action_from(self, &[imat(t as i64, 0, 0, 1)]))
    }

    /// The integral matrices T_t g_i whose summed action IS the raising map
    /// delta_t (exposed for the generator-by-generator well-definedness
    /// re-verification in the tests).
    pub(crate) fn degeneracy_raising_integral_mats(
        &self,
        lower: &ModularSymbolsGamma0,
        t: u64,
    ) -> Result<Vec<IMat>, String> {
        let (n, m) = (self.level(), lower.level());
        if m == 0 || !n.is_multiple_of(m) {
            return Err(format!("{m} does not divide the level {n}"));
        }
        if t == 0 || !(n / m).is_multiple_of(t) {
            return Err(format!("t = {t} does not divide N/M = {}", n / m));
        }
        let reps = gamma_prime_coset_reps(lower, n, t)?;
        let tt = imat(1, 0, 0, t as i64);
        Ok(reps.iter().map(|g| imat_mul(&tt, g)).collect())
    }

    /// The degeneracy RAISING map delta_t : lower -> self (the transfer
    /// composed with [[1,0],[0,t]]; see the module docs for the derivation
    /// and the exact composition laws that certify it).  Returns the matrix
    /// with columns indexed by the basis of `lower` and rows by the basis
    /// of `self`.
    pub fn degeneracy_raising_matrix(
        &self,
        lower: &ModularSymbolsGamma0,
        t: u64,
    ) -> Result<Matrix<Rational>, String> {
        let mats = self.degeneracy_raising_integral_mats(lower, t)?;
        Ok(self.gl2_sum_action_from(lower, &mats))
    }

    /// Basis of the NEW cuspidal subspace (intersection of the kernels of
    /// all lowering maps pi_1, pi_p to the levels N/p for primes p | N),
    /// as coefficient vectors with respect to [`Self::cuspidal_basis`].
    pub fn cuspidal_new_subspace(&self) -> Result<Vec<Vec<Rational>>, String> {
        let n = self.level();
        let s = self.cuspidal_dimension();
        if s == 0 {
            return Ok(Vec::new());
        }
        // stack the images of the cuspidal basis under every lowering map
        let mut stacked_rows: Vec<Vec<Rational>> = vec![Vec::new(); s];
        for p in prime_divisors(n) {
            let lower = ModularSymbolsGamma0::new(n / p);
            if lower.dimension() == 0 {
                continue;
            }
            for t in [1u64, p] {
                let l = self.degeneracy_lowering_matrix(&lower, t)?;
                for (k, v) in self.cuspidal_basis().iter().enumerate() {
                    // image = L * v
                    for i in 0..lower.dimension() {
                        let mut sum = Rational::zero();
                        for (j, x) in v.iter().enumerate() {
                            if x.is_zero() {
                                continue;
                            }
                            sum = &sum + &(l.get(i, j).expect("entry in range") * x);
                        }
                        stacked_rows[k].push(sum);
                    }
                }
            }
        }
        let total = stacked_rows[0].len();
        if total == 0 {
            // no proper level with nonzero space: everything is new
            return Ok((0..s)
                .map(|i| {
                    let mut v = vec![Rational::zero(); s];
                    v[i] = Rational::one();
                    v
                })
                .collect());
        }
        // kernel of the (total x s) matrix whose column k is the stacked
        // image of the k-th cuspidal basis vector
        let mut flat = Vec::with_capacity(total * s);
        for i in 0..total {
            for row in stacked_rows.iter() {
                flat.push(row[i].clone());
            }
        }
        let b = Matrix::from_vec(total, s, flat).expect("stacked matrix shape");
        b.kernel().map_err(|e| format!("kernel failed: {e:?}"))
    }

    /// Whether a summand of the cuspidal Hecke decomposition is NEW
    /// (contained in the kernel of every lowering map) or OLD (intersecting
    /// the new subspace trivially).  Errors honestly if the summand is
    /// neither, which would mean the Hecke refinement failed to separate an
    /// old block from a new one.
    pub fn summand_is_new(&self, summand: &HeckeSummand) -> Result<bool, String> {
        let n = self.level();
        let dim = summand.dimension();
        if dim == 0 {
            return Err("empty summand".to_string());
        }
        // stack the lowering images of the summand's basis vectors
        let mut stacked: Vec<Vec<Rational>> = vec![Vec::new(); dim];
        for p in prime_divisors(n) {
            let lower = ModularSymbolsGamma0::new(n / p);
            if lower.dimension() == 0 {
                continue;
            }
            for t in [1u64, p] {
                let l = self.degeneracy_lowering_matrix(&lower, t)?;
                for (k, v) in summand.ambient_basis().iter().enumerate() {
                    for i in 0..lower.dimension() {
                        let mut sum = Rational::zero();
                        for (j, x) in v.iter().enumerate() {
                            if x.is_zero() {
                                continue;
                            }
                            sum = &sum + &(l.get(i, j).expect("entry in range") * x);
                        }
                        stacked[k].push(sum);
                    }
                }
            }
        }
        let total = stacked[0].len();
        if total == 0 {
            return Ok(true); // no lower level: everything is new
        }
        let mut flat = Vec::with_capacity(total * dim);
        for i in 0..total {
            for row in stacked.iter() {
                flat.push(row[i].clone());
            }
        }
        let b = Matrix::from_vec(total, dim, flat).expect("stacked matrix shape");
        let ker = b.kernel().map_err(|e| format!("kernel failed: {e:?}"))?;
        match ker.len() {
            k if k == dim => Ok(true),
            0 => Ok(false),
            k => Err(format!(
                "summand meets the new subspace in dimension {k}, neither 0 nor {dim}: \
                 the Hecke refinement failed to separate old from new"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modsym::decomposition::lin_comb;
    use rustmath_matrix::charpoly_berkowitz;
    use rustmath_polynomials::UnivariatePolynomial;

    fn rat(k: i64) -> Rational {
        Rational::from_integer(Integer::from(k))
    }

    /// Monic polynomial from ascending integer coefficients.
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

    /// Rank of the column collection (each column a vector of length dim).
    fn rank(cols: &[Vec<Rational>], dim: usize) -> usize {
        if cols.is_empty() {
            return 0;
        }
        let mut flat = Vec::with_capacity(dim * cols.len());
        for i in 0..dim {
            for c in cols {
                flat.push(c[i].clone());
            }
        }
        let m = Matrix::from_vec(dim, cols.len(), flat).expect("shape");
        m.reduced_row_echelon_form()
            .expect("exact rref over Q cannot fail")
            .pivots
            .len()
    }

    /// GATE: the degeneracy maps are WELL DEFINED on the Manin-symbol
    /// quotient: for EVERY generator of the source space (not only the
    /// basis) the direct sum-of-matrices image equals the matrix applied to
    /// the generator's projection.  This is exactly the statement that the
    /// generator-level formula descends to the quotient (raising: the
    /// transfer sum sum_i T_t g_i x; lowering: [[t,0],[0,1]] x).
    #[test]
    fn test_degeneracy_maps_well_defined_on_all_generators() {
        let m11 = ModularSymbolsGamma0::new(11);
        // transfer coset counts psi(N)/psi(11): psi(22)/psi(11) = 36/12 = 3,
        // psi(33)/psi(11) = 48/12 = 4
        for (n, ts, ncosets) in [(22u64, [1u64, 2], 3usize), (33, [1, 3], 4)] {
            let big = ModularSymbolsGamma0::new(n);
            for &t in &ts {
                let mats = big.degeneracy_raising_integral_mats(&m11, t).unwrap();
                assert_eq!(mats.len(), ncosets, "psi({n})/psi(11) transfer cosets");
                let d = big.degeneracy_raising_matrix(&m11, t).unwrap();
                for i in 0..m11.num_generators() {
                    let direct = big.gl2_sum_image_of_source_generator(&m11, &mats, i);
                    let via = mv(&d, m11.manin_generator_coords(i));
                    assert_eq!(direct, via, "raising delta_{t} to {n}, generator {i}");
                }
                let lmats = [imat(t as i64, 0, 0, 1)];
                let l = big.degeneracy_lowering_matrix(&m11, t).unwrap();
                for i in 0..big.num_generators() {
                    let direct = m11.gl2_sum_image_of_source_generator(&big, &lmats, i);
                    let via = mv(&l, big.manin_generator_coords(i));
                    assert_eq!(direct, via, "lowering pi_{t} from {n}, generator {i}");
                }
            }
        }
    }

    /// GATE: the exact composition laws that certify the transfer
    /// construction (derived in the module docs, no external inputs):
    ///   pi_1 delta_1 = (index) id   because each transfer term g_i x is
    ///       Gamma0(M)-equivalent to x at level M;
    ///   pi_t delta_t = (index) id   because D_t T_t = t * identity acts
    ///       trivially on weight-2 symbols;
    ///   pi_1 delta_p = pi_p delta_1 = T_p at level M (p prime, p coprime
    ///       to M): the p+1 = index summed det-p matrices hit the p+1 left
    ///       cosets of the T_p double coset.
    /// The index is psi(N)/psi(11) = 3 for N = 22 and 4 for N = 33.
    #[test]
    fn test_composition_laws_certify_transfer() {
        let m11 = ModularSymbolsGamma0::new(11);
        let id3: Matrix<Rational> = Matrix::identity(m11.dimension());
        for (n, p, index) in [(22u64, 2u64, 3i64), (33, 3, 4)] {
            let big = ModularSymbolsGamma0::new(n);
            let d1 = big.degeneracy_raising_matrix(&m11, 1).unwrap();
            let dp = big.degeneracy_raising_matrix(&m11, p).unwrap();
            let p1 = big.degeneracy_lowering_matrix(&m11, 1).unwrap();
            let pp = big.degeneracy_lowering_matrix(&m11, p).unwrap();
            assert_eq!(
                mm(&p1, &d1),
                id3.scalar_mul(&rat(index)),
                "pi_1 delta_1 at {n}"
            );
            assert_eq!(
                mm(&pp, &dp),
                id3.scalar_mul(&rat(index)),
                "pi_p delta_p at {n}"
            );
            let tp = m11.hecke_matrix(p);
            assert_eq!(mm(&p1, &dp), tp, "pi_1 delta_{p} = T_{p}(11) via {n}");
            assert_eq!(mm(&pp, &d1), tp, "pi_{p} delta_1 = T_{p}(11) via {n}");
        }
    }

    /// GATE: Hecke equivariance T_q delta_t = delta_t T_q and
    /// T_q pi_t = pi_t T_q for q coprime to the big level, as exact
    /// ambient matrix identities.
    #[test]
    fn test_degeneracy_hecke_equivariance() {
        let m11 = ModularSymbolsGamma0::new(11);
        for (n, ts, qs) in [
            (22u64, vec![1u64, 2], vec![3u64, 5, 7, 13]),
            (33, vec![1, 3], vec![2, 5]),
        ] {
            let big = ModularSymbolsGamma0::new(n);
            for &t in &ts {
                let d = big.degeneracy_raising_matrix(&m11, t).unwrap();
                let l = big.degeneracy_lowering_matrix(&m11, t).unwrap();
                for &q in &qs {
                    let tq_big = big.hecke_matrix(q);
                    let tq_low = m11.hecke_matrix(q);
                    assert_eq!(
                        mm(&tq_big, &d),
                        mm(&d, &tq_low),
                        "T_{q} delta_{t} != delta_{t} T_{q} at {n}"
                    );
                    assert_eq!(
                        mm(&tq_low, &l),
                        mm(&l, &tq_big),
                        "T_{q} pi_{t} != pi_{t} T_{q} at {n}"
                    );
                }
            }
        }
    }

    /// GATE (the level-22 old/new seam, RESOLVED): the two degeneracy
    /// images of the level-11 cuspidal space span the whole cuspidal space
    /// at 22 (new space = 0), and U_2 on that old space has EXACTLY the
    /// block structure predicted by the degeneracy theory from the
    /// point-counted a_2(11a) = -2:
    ///   * U_2 delta_1 = 2 delta_2 as a GLOBAL matrix identity;
    ///   * in the ordered basis (delta_1 v1, delta_1 v2, delta_2 v1,
    ///     delta_2 v2) of the old cuspidal space the restriction is the
    ///     2x2-scalar-block matrix [[0, -I2], [2 I2, -2 I2]], conjugate to
    ///     the classical [[a_2, 1], [-2, 0]] per sign copy;
    ///   * minimal polynomial: U_2^2 + 2 U_2 + 2 = 0 on the old space
    ///     (x^2 - a_2 x + 2 with a_2 = -2) - this EXPLAINS the honest
    ///     algebraic eigenvalue x^2 + 2x + 2 of stage 1 entirely through
    ///     level-11 data;
    ///   * charpoly (x^2 + 2x + 2)^2; U_11 = a_11 * id = id (point-counted
    ///     a_11 = 1);
    ///   * the Atkin-Lehner W_11 at level 22 INTERTWINES the degeneracy
    ///     maps (W_11^(22)) delta_t = delta_t (W_11^(11)) globally, the
    ///     classical commutation of W_Q with level raising for gcd(Q,t)=1.
    #[test]
    fn test_level_22_old_space_split_resolved() {
        let m11 = ModularSymbolsGamma0::new(11);
        let m22 = ModularSymbolsGamma0::new(22);
        let d1 = m22.degeneracy_raising_matrix(&m11, 1).unwrap();
        let d2 = m22.degeneracy_raising_matrix(&m11, 2).unwrap();
        // old cuspidal basis: images of the level-11 cuspidal basis
        let mut old: Vec<Vec<Rational>> = Vec::new();
        for d in [&d1, &d2] {
            for v in m11.cuspidal_basis() {
                old.push(mv(d, v));
            }
        }
        for (k, v) in old.iter().enumerate() {
            assert!(m22.is_cuspidal(v), "old basis vector {k} must be cuspidal");
        }
        assert_eq!(
            rank(&old, m22.dimension()),
            4,
            "old space fills the 2g = 4 cuspidal dims"
        );
        // global identity U_2 delta_1 = 2 delta_2
        let u2 = m22.hecke_matrix(2);
        assert_eq!(
            mm(&u2, &d1),
            d2.scalar_mul(&rat(2)),
            "U_2 delta_1 = 2 delta_2 globally"
        );
        // restriction to the old space in the (delta_1 v, delta_2 v) basis
        let r = crate::modsym::decomposition::restrict_to_column_span(&u2, &old).unwrap();
        let expected: Vec<i64> = vec![
            0, 0, -1, 0, //
            0, 0, 0, -1, //
            2, 0, -2, 0, //
            0, 2, 0, -2,
        ];
        let expected = Matrix::from_vec(4, 4, expected.into_iter().map(rat).collect()).unwrap();
        assert_eq!(r, expected, "U_2 block structure on the 22-old space");
        // minimal polynomial x^2 + 2x + 2 (= x^2 - a_2 x + 2, a_2 = -2)
        let min = ((mm(&r, &r) + r.scalar_mul(&rat(2))).unwrap()
            + Matrix::identity(4).scalar_mul(&rat(2)))
        .unwrap();
        assert_eq!(
            min,
            Matrix::zeros(4, 4),
            "U_2^2 + 2 U_2 + 2 = 0 on the old space"
        );
        assert_eq!(
            charpoly_berkowitz(&r).unwrap(),
            poly(&[2, 2, 1]) * poly(&[2, 2, 1]),
            "charpoly (x^2 + 2x + 2)^2"
        );
        // U_11 = a_11 * id with the point-counted a_11(11a) = 1
        let u11 =
            crate::modsym::decomposition::restrict_to_column_span(&m22.hecke_matrix(11), &old)
                .unwrap();
        assert_eq!(u11, Matrix::identity(4), "U_11 = id on the old space");
        // W_11 intertwines the degeneracy maps
        let w11_22 = m22.atkin_lehner_matrix(11).unwrap();
        let w11_11 = m11.atkin_lehner_matrix(11).unwrap();
        for (t, d) in [(1u64, &d1), (2, &d2)] {
            assert_eq!(
                mm(&w11_22, d),
                mm(d, &w11_11),
                "W_11(22) delta_{t} = delta_{t} W_11(11)"
            );
        }
    }

    /// GATE (level 44, resolves the stage-1 Mixed deferral): the three
    /// degeneracy images (t = 1, 2, 4) of the level-11 cuspidal space span
    /// exactly the 6-dimensional old summand of the decomposition, and U_2
    /// on it is the predicted shift-with-relation block matrix: globally
    /// U_2 delta_1 = 2 delta_2 and U_2 delta_2 = 2 delta_4, and on the old
    /// cuspidal space U_2 delta_4 = -delta_2 - 2 delta_4 (the a_2 = -2
    /// relation entering at the top of the tower), so
    /// charpoly = x^2 (x^2 + 2x + 2)^2: EXACTLY the stage-1 Mixed factors
    /// {x, x^2 + 2x + 2} with multiplicity 2, now derived rather than
    /// merely observed.
    #[test]
    fn test_level_44_old_block_resolves_stage1_mixed() {
        let m11 = ModularSymbolsGamma0::new(11);
        let m44 = ModularSymbolsGamma0::new(44);
        let d1 = m44.degeneracy_raising_matrix(&m11, 1).unwrap();
        let d2 = m44.degeneracy_raising_matrix(&m11, 2).unwrap();
        let d4 = m44.degeneracy_raising_matrix(&m11, 4).unwrap();
        let mut old: Vec<Vec<Rational>> = Vec::new();
        for d in [&d1, &d2, &d4] {
            for v in m11.cuspidal_basis() {
                old.push(mv(d, v));
            }
        }
        for v in &old {
            assert!(m44.is_cuspidal(v));
        }
        assert_eq!(
            rank(&old, m44.dimension()),
            6,
            "three independent images of dim 2"
        );
        // the old span IS the 6-dim summand of the stage-1 decomposition
        let dec = m44.cuspidal_hecke_decomposition().unwrap();
        let w6 = &dec.summands()[1];
        assert_eq!(w6.dimension(), 6);
        let mut combined = old.clone();
        combined.extend(w6.ambient_basis().iter().cloned());
        assert_eq!(
            rank(&combined, m44.dimension()),
            6,
            "degeneracy images span the same subspace as the 6-dim Hecke summand"
        );
        // global shift identities
        let u2 = m44.hecke_matrix(2);
        assert_eq!(
            mm(&u2, &d1),
            d2.scalar_mul(&rat(2)),
            "U_2 delta_1 = 2 delta_2"
        );
        assert_eq!(
            mm(&u2, &d2),
            d4.scalar_mul(&rat(2)),
            "U_2 delta_2 = 2 delta_4"
        );
        // block structure on the old space
        let r = crate::modsym::decomposition::restrict_to_column_span(&u2, &old).unwrap();
        let expected: Vec<i64> = vec![
            0, 0, 0, 0, 0, 0, //
            0, 0, 0, 0, 0, 0, //
            2, 0, 0, 0, -1, 0, //
            0, 2, 0, 0, 0, -1, //
            0, 0, 2, 0, -2, 0, //
            0, 0, 0, 2, 0, -2,
        ];
        let expected = Matrix::from_vec(6, 6, expected.into_iter().map(rat).collect()).unwrap();
        assert_eq!(r, expected, "U_2 blocks on the 44-old space");
        assert_eq!(
            charpoly_berkowitz(&r).unwrap(),
            poly(&[0, 1]) * poly(&[0, 1]) * poly(&[2, 2, 1]) * poly(&[2, 2, 1]),
            "charpoly x^2 (x^2 + 2x + 2)^2 = the stage-1 Mixed factors"
        );
        // U_11 = a_11 = 1 on the old block
        let u11 =
            crate::modsym::decomposition::restrict_to_column_span(&m44.hecke_matrix(11), &old)
                .unwrap();
        assert_eq!(u11, Matrix::identity(6));
    }

    /// GATE: dimensions of the NEW cuspidal subspace, each derived
    /// independently from the genus table and the multiplicity-one count of
    /// lower-level images: dim new = 2g(N) - sum over proper M | N of
    /// sigma_0(N/M) * (dim new at M):
    ///   11, 23, 37 prime: everything is new (2g = 2, 4, 4);
    ///   22: old images of 11a fill 2 * 2 = 4 = 2g, new = 0;
    ///   33: 2g = 6, old-11a = 4, new = 2 (the newform 33a);
    ///   44: 2g = 8, old-11a = 6 (t = 1, 2, 4), new = 2 (44a).
    #[test]
    fn test_new_subspace_dimensions() {
        for (n, expected) in [(11u64, 2usize), (22, 0), (23, 4), (33, 2), (37, 4), (44, 2)] {
            let m = ModularSymbolsGamma0::new(n);
            let new = m.cuspidal_new_subspace().unwrap();
            assert_eq!(new.len(), expected, "dim new cuspidal at level {n}");
            // every new vector really is cuspidal and killed by all lowering maps
            let cb = m.cuspidal_basis();
            for x in &new {
                let v = lin_comb(cb, x);
                assert!(m.is_cuspidal(&v));
            }
        }
    }

    /// GATE: old/new classification of the Hecke summands, plus the
    /// complementarity old + new = full cuspidal space at 33.
    #[test]
    fn test_summand_old_new_classification() {
        // prime levels: every summand is new
        for n in [11u64, 23, 37] {
            let m = ModularSymbolsGamma0::new(n);
            for w in m.cuspidal_hecke_decomposition().unwrap().summands() {
                assert_eq!(m.summand_is_new(w), Ok(true), "summand at prime level {n}");
            }
        }
        // 22: the single 4-dim summand is old
        let m22 = ModularSymbolsGamma0::new(22);
        let dec22 = m22.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(dec22.summands().len(), 1);
        assert_eq!(m22.summand_is_new(&dec22.summands()[0]), Ok(false));
        // 44: 2-dim newform summand + 6-dim old block
        let m44 = ModularSymbolsGamma0::new(44);
        let dec44 = m44.cuspidal_hecke_decomposition().unwrap();
        let dims: Vec<usize> = dec44
            .summands()
            .iter()
            .map(HeckeSummand::dimension)
            .collect();
        assert_eq!(dims, vec![2, 6]);
        assert_eq!(
            m44.summand_is_new(&dec44.summands()[0]),
            Ok(true),
            "44a is new"
        );
        assert_eq!(
            m44.summand_is_new(&dec44.summands()[1]),
            Ok(false),
            "old 11a block"
        );
        // 45 = 9 * 5: 2-dim newform 45a + 4-dim old block of 15a
        let m45 = ModularSymbolsGamma0::new(45);
        let dec45 = m45.cuspidal_hecke_decomposition().unwrap();
        let dims: Vec<usize> = dec45
            .summands()
            .iter()
            .map(HeckeSummand::dimension)
            .collect();
        assert_eq!(dims, vec![2, 4]);
        assert_eq!(
            m45.summand_is_new(&dec45.summands()[0]),
            Ok(true),
            "45a is new"
        );
        assert_eq!(
            m45.summand_is_new(&dec45.summands()[1]),
            Ok(false),
            "old 15a block"
        );
        // 33: old images + new subspace together span the cuspidal space
        let m11 = ModularSymbolsGamma0::new(11);
        let m33 = ModularSymbolsGamma0::new(33);
        let mut cols: Vec<Vec<Rational>> = Vec::new();
        for t in [1u64, 3] {
            let d = m33.degeneracy_raising_matrix(&m11, t).unwrap();
            for v in m11.cuspidal_basis() {
                cols.push(mv(&d, v));
            }
        }
        assert_eq!(rank(&cols, m33.dimension()), 4, "old space at 33 has dim 4");
        let cb33 = m33.cuspidal_basis();
        for x in m33.cuspidal_new_subspace().unwrap() {
            cols.push(lin_comb(cb33, &x));
        }
        assert_eq!(
            rank(&cols, m33.dimension()),
            6,
            "old + new = full cuspidal space (2g = 6) at 33"
        );
    }

    /// Honest errors for invalid degeneracy parameters.
    #[test]
    fn test_degeneracy_error_cases() {
        let m11 = ModularSymbolsGamma0::new(11);
        let m22 = ModularSymbolsGamma0::new(22);
        let m23 = ModularSymbolsGamma0::new(23);
        assert!(
            m23.degeneracy_raising_matrix(&m11, 1).is_err(),
            "11 does not divide 23"
        );
        assert!(m23.degeneracy_lowering_matrix(&m11, 1).is_err());
        assert!(
            m22.degeneracy_raising_matrix(&m11, 4).is_err(),
            "4 does not divide 22/11"
        );
        assert!(
            m22.degeneracy_lowering_matrix(&m11, 0).is_err(),
            "t = 0 rejected"
        );
    }
}
