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
//! # Labelling a Hecke summand New / Old(M), and what U_l can and cannot do
//!
//! Write Im(M) = sum over t | N/M of delta_t(S_2(Gamma0(M))), the span at
//! level N of ALL degeneracy images from level M.  Two facts drive the
//! labelling ([`ModularSymbolsGamma0::summand_source_level`]):
//!
//! * Im is MONOTONE: M' | M | N implies Im(M') is contained in Im(M).  Proof:
//!   Im(M) contains delta_u(delta_s(S_2(M'))) = delta_{us}(S_2(M')) for every
//!   u | N/M and s | M/M', and every t | N/M' factors as t = u s that way
//!   (for each prime take u-exponent min(v_p(t), v_p(N/M))).  So the set of
//!   levels whose image contains a given summand is closed UPWARD under
//!   divisibility, and scanning the divisors of N in increasing numeric order
//!   and taking the first hit returns the MINIMAL such level: any proper
//!   divisor of it is numerically smaller and was already rejected.
//! * For a newform f of exact level M_0, the f-isotypic part of Im(M) is zero
//!   unless M_0 | M (f simply does not occur in S_2(Gamma0(M))).  So for a
//!   summand carrying one newform eigensystem the set of levels whose image
//!   contains it is EXACTLY the multiples of M_0 dividing N, and the minimal
//!   one is the newform's own level.  Level 66 is the sharp test: the 11a
//!   block lies inside Im(11), Im(22), Im(33) and Im(66) alike, and only the
//!   minimality makes the answer 11 rather than 22 or 33.
//!
//! The multiplicity is then recomputed by a SECOND, disjoint route - the
//! LOWERING maps: pi_1 restricted to the summand W = sum_t delta_t(V_M) has
//! image exactly V_M (pi_1 delta_1 = index * id gives one inclusion, Hecke
//! stability of pi_1(W) with the same eigensystem gives the other), so
//! multiplicity = dim W / rank(pi_1 | W), and this must come out equal to
//! sigma_0(N/M).  Raising and lowering are then cross-certifying each other.
//!
//! ## What U_l can split over Q, and what it provably cannot
//!
//! Let f be a newform of level M and l a prime with l | N/M.  On the plane
//! spanned by delta_1(f), delta_l(f) the classical matrices are
//!
//! ```text
//! l does NOT divide M:  U_l = [[a_l, 1], [-l, 0]],  charpoly x^2 - a_l x + l
//! l DOES divide M:      U_l = [[a_l, 1], [ 0, 0]],  charpoly x^2 - a_l x
//! ```
//!
//! * **l does not divide M: the two copies can NEVER be separated over Q.**
//!   A rational root lambda of x^2 - a_l x + l would give a_l = lambda + l/lambda,
//!   so |a_l| >= 2 sqrt(l) by AM-GM, with equality only for lambda = +-sqrt(l),
//!   irrational.  Hence |a_l| > 2 sqrt(l) strictly, contradicting the
//!   Ramanujan-Petersson bound |a_l| <= 2 sqrt(l) (Eichler-Shimura/Hasse for
//!   weight 2).  Equivalently, for a rational newform the discriminant
//!   a_l^2 - 4l is strictly negative (it is <= 0 by Ramanujan and = 0 would
//!   force l to be a perfect square).  This is a THEOREM, not a limitation of
//!   the implementation: the honest deliverable for such a block is the
//!   irreducible quadratic obstruction itself, reported by
//!   [`UlRefinement::obstructions`].  It is verified here at N = 22 (a_2 = -2,
//!   x^2 + 2x + 2, disc -4), 42 and 66, and the general discriminant claim is
//!   re-checked in [`ModularSymbolsGamma0::u_refinement_of_summand`]'s tests.
//! * **l divides M: the plane DOES split over Q.**  charpoly x(x - a_l) has
//!   the two rational roots 0 and a_l, and a_l = +-1 for l || M, so the roots
//!   are distinct and the refinement is genuine (N = 45, M = 15, l = 3).
//! * **l^2 | N/M: a partial split.**  The images t | l^v (v = v_l(N/M) >= 2)
//!   span a (v+1)-dimensional space on which charpoly(U_l) =
//!   x^(v-1) (x^2 - a_l x + l): the nilpotent part splits off over Q, but the
//!   quadratic factor remains irreducible (N = 44, v = 2, x^2 (x^2 + 2x + 2)).
//!
//! So [`SummandHeckeAction::Mixed`] stops being a confession of defeat and
//! becomes a structure theorem: [`ModularSymbolsGamma0::u_refinement_of_summand`]
//! performs the refinement wherever it exists and returns the irreducible
//! factor - the exact obstruction - wherever it does not.
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

use super::decomposition::{
    eval_poly_at_matrix, factor_monic_rational_certified, is_prime_u64, lin_comb, mat_pow,
    restrict_to_column_span, CuspidalHeckeDecomposition, HeckeSummand,
};
use super::gamma0::ModularSymbolsGamma0;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::{charpoly_berkowitz, Matrix};
use rustmath_polynomials::UnivariatePolynomial;
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

/// The divisors of n, in INCREASING order (the order is what makes the
/// source-level scan return the minimal level; see the module docs).
fn divisors(n: u64) -> Vec<u64> {
    (1..=n).filter(|d| n.is_multiple_of(*d)).collect()
}

/// sigma_0(n), the number of divisors.
fn sigma0(n: u64) -> u32 {
    divisors(n).len() as u32
}

/// Rank of a collection of column vectors, each of length `dim`.
fn column_rank(cols: &[Vec<Rational>], dim: usize) -> Result<usize, String> {
    if cols.is_empty() || dim == 0 {
        return Ok(0);
    }
    let mut flat = Vec::with_capacity(dim * cols.len());
    for i in 0..dim {
        for c in cols {
            flat.push(c[i].clone());
        }
    }
    let m = Matrix::from_vec(dim, cols.len(), flat).expect("column collection shape");
    m.rank().map_err(|e| format!("rank failed: {e:?}"))
}

/// Whether every column of `sub` lies in the span of the columns of `sup`
/// (exactly, over Q): true iff adjoining `sub` does not raise the rank.
fn span_contains(sup: &[Vec<Rational>], sub: &[Vec<Rational>], dim: usize) -> Result<bool, String> {
    let r = column_rank(sup, dim)?;
    let mut both = sup.to_vec();
    both.extend_from_slice(sub);
    Ok(column_rank(&both, dim)? == r)
}

/// Image of a vector (in the coordinates the matrix acts on) under a matrix.
fn apply(m: &Matrix<Rational>, v: &[Rational]) -> Vec<Rational> {
    (0..m.rows())
        .map(|i| {
            let mut sum = Rational::zero();
            for (j, x) in v.iter().enumerate() {
                if x.is_zero() {
                    continue;
                }
                sum = &sum + &(m.get(i, j).expect("entry in range") * x);
            }
            sum
        })
        .collect()
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

    /// Express cuspidal vectors given in AMBIENT coordinates in coordinates
    /// with respect to [`Self::cuspidal_basis`].  Errors honestly if an input
    /// is not in the cuspidal span (which for a degeneracy image would mean
    /// the raising map is broken, not that the caller misused the function).
    ///
    /// Works by row reducing the augmented matrix [C | V] whose first block is
    /// the cuspidal basis: since C has full column rank s, the rref has exactly
    /// the pivots 0..s in the C-block, the coefficient columns are read off the
    /// first s rows of the V-block, and ANY pivot in the V-block certifies that
    /// some input is outside the cuspidal subspace.
    fn cuspidal_coordinates(&self, cols: &[Vec<Rational>]) -> Result<Vec<Vec<Rational>>, String> {
        if cols.is_empty() {
            return Ok(Vec::new());
        }
        let s = self.cuspidal_dimension();
        let dim = self.dimension();
        let k = cols.len();
        let mut flat = Vec::with_capacity(dim * (s + k));
        for i in 0..dim {
            for c in self.cuspidal_basis() {
                flat.push(c[i].clone());
            }
            for v in cols {
                flat.push(v[i].clone());
            }
        }
        let aug = Matrix::from_vec(dim, s + k, flat).expect("augmented shape");
        let rref = aug
            .reduced_row_echelon_form()
            .map_err(|e| format!("rref failed: {e:?}"))?;
        if !rref.pivots.iter().copied().take(s).eq(0..s) {
            return Err(format!(
                "the cuspidal basis at level {} is not of full column rank",
                self.level()
            ));
        }
        if rref.pivots.len() > s {
            return Err(format!(
                "a vector at level {} is not in the cuspidal subspace",
                self.level()
            ));
        }
        (0..k)
            .map(|j| {
                (0..s)
                    .map(|r| {
                        rref.matrix
                            .get(r, s + j)
                            .cloned()
                            .map_err(|e| format!("entry out of range: {e:?}"))
                    })
                    .collect()
            })
            .collect()
    }

    /// Im(M): a basis, in AMBIENT quotient-basis coordinates, of the span of
    /// ALL degeneracy raising images delta_t(S_2(Gamma0(M))) for t | (N/M).
    ///
    /// This is the single primitive behind both the old subspace
    /// ([`Self::cuspidal_old_subspace`], the union over the proper divisors)
    /// and the old/new labelling ([`Self::summand_source_level`], the minimal
    /// M whose image contains the summand), so the two can never disagree.
    ///
    /// For M = N the only t is 1 and delta_1 is the transfer along the index-1
    /// inclusion Gamma0(N) < Gamma0(N), i.e. the identity: Im(N) is the whole
    /// cuspidal subspace.  That is not special-cased; it falls out of the
    /// general construction (the tests check delta_1^(N <- N) = id exactly),
    /// which is what lets a NEW summand be labelled by the very same scan that
    /// labels an old one.
    pub fn degeneracy_image_from_level(&self, m: u64) -> Result<Vec<Vec<Rational>>, String> {
        let n = self.level();
        if m == 0 || !n.is_multiple_of(m) {
            return Err(format!("{m} does not divide the level {n}"));
        }
        let lower = ModularSymbolsGamma0::new(m);
        if lower.cuspidal_dimension() == 0 {
            return Ok(Vec::new()); // a genus-0 level contributes nothing
        }
        let mut images: Vec<Vec<Rational>> = Vec::new();
        for t in divisors(n / m) {
            let d = self.degeneracy_raising_matrix(&lower, t)?;
            for v in lower.cuspidal_basis() {
                images.push(apply(&d, v));
            }
        }
        let dim = self.dimension();
        let mut flat = Vec::with_capacity(dim * images.len());
        for i in 0..dim {
            for c in &images {
                flat.push(c[i].clone());
            }
        }
        let mat = Matrix::from_vec(dim, images.len(), flat).expect("image matrix shape");
        mat.image().map_err(|e| format!("image failed: {e:?}"))
    }

    /// The OLD cuspidal subspace of S_2(Gamma0(N)): the span of all degeneracy
    /// raising images delta_t(S_2(Gamma0(M))) for every M | N with M < N and
    /// every t | (N/M), as coefficient vectors with respect to
    /// [`Self::cuspidal_basis`] (the same convention as
    /// [`Self::cuspidal_new_subspace`]).
    ///
    /// The sum runs over ALL proper divisors M, not only the maximal ones:
    /// that is redundant (the raising maps compose, so the maximal M already
    /// span the whole old space) but it needs no argument to be believed, and
    /// the redundant generators cost nothing once the span is taken.
    ///
    /// Certified by dim(new) + dim(old) = dim(cuspidal) in the tests.
    pub fn cuspidal_old_subspace(&self) -> Result<Vec<Vec<Rational>>, String> {
        let n = self.level();
        let s = self.cuspidal_dimension();
        if s == 0 {
            return Ok(Vec::new());
        }
        let mut images: Vec<Vec<Rational>> = Vec::new();
        for m in divisors(n) {
            if m == n {
                continue;
            }
            images.extend(self.degeneracy_image_from_level(m)?);
        }
        if images.is_empty() {
            return Ok(Vec::new());
        }
        // the raising maps send cuspidal to cuspidal, so this cannot fail
        // unless the construction is wrong - which is exactly what we want to
        // hear about
        let coords = self.cuspidal_coordinates(&images)?;
        let mut flat = Vec::with_capacity(s * coords.len());
        for i in 0..s {
            for c in &coords {
                flat.push(c[i].clone());
            }
        }
        let mat = Matrix::from_vec(s, coords.len(), flat).expect("coefficient matrix shape");
        mat.image().map_err(|e| format!("image failed: {e:?}"))
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

    /// The MINIMAL level M | N such that the summand lies inside Im(M), the
    /// span of the degeneracy raising images from level M
    /// ([`Self::degeneracy_image_from_level`]).  For a summand carrying a
    /// single newform eigensystem this IS the newform's own level; M = N means
    /// the summand is new.
    ///
    /// The determination is honest linear algebra - subspace containment by
    /// rank comparison, not a guess from dimensions - and minimality is
    /// structural rather than asserted: the divisors are scanned in increasing
    /// numeric order, so when M is accepted every proper divisor of M has
    /// already been rejected.  See the module docs for why Im is monotone in M
    /// (which is what makes "minimal" the right question) and why level 66,
    /// where the 11a block sits inside Im(11), Im(22), Im(33) and Im(66)
    /// alike, is the test that a non-minimal implementation fails.
    ///
    /// Works for any Hecke-stable subspace.  The stronger
    /// [`Self::summand_provenance`] additionally re-derives the multiplicity
    /// through the lowering maps and demands sigma_0(N/M); use this one when
    /// the input is a proper piece of a summand (e.g. a U_l refinement), for
    /// which no sigma_0 law holds.
    pub fn summand_source_level(&self, summand: &HeckeSummand) -> Result<u64, String> {
        let n = self.level();
        if summand.dimension() == 0 {
            return Err("empty summand has no source level".to_string());
        }
        let dim = self.dimension();
        for m in divisors(n) {
            let img = self.degeneracy_image_from_level(m)?;
            if img.is_empty() {
                continue;
            }
            if span_contains(&img, summand.ambient_basis(), dim)? {
                return Ok(m);
            }
        }
        // Im(N) is the whole cuspidal subspace and every summand is cuspidal,
        // so the scan cannot fall through unless the summand is not cuspidal.
        Err(format!(
            "the summand is contained in no degeneracy image, not even Im({n}) = the \
             cuspidal subspace itself: it is not a cuspidal Hecke summand of this space"
        ))
    }

    /// The number of degeneracy copies of the source level's newform block that
    /// the summand contains, computed through the LOWERING maps only: pi_1
    /// restricted to W = sum_{t | N/M} delta_t(V_M) has image exactly V_M, so
    /// multiplicity = dim(W) / rank(pi_1 | W).  For M < N this is disjoint from
    /// the raising-map computation that found M, which is what makes the
    /// `== sigma_0(N/M)` check in [`Self::summand_provenance`] a real
    /// cross-certification of lowering against raising.
    ///
    /// EXACTLY ONE CASE IS EXEMPT, and it is the New one: when M == N the routine
    /// short-circuits to 1 without touching a lowering map.  It is entitled to --
    /// pi_1 is then the identity, so the general computation would return
    /// dim(W)/dim(W) = 1 anyway -- but it does mean that for a NEW summand the
    /// `== sigma_0(N/M)` check in `summand_provenance` compares 1 against
    /// sigma_0(1) = 1 and certifies nothing.  The New label rests entirely on the
    /// minimality of the source level found by [`Self::summand_source_level`],
    /// i.e. on the raising-map scan; the multiplicity cross-check adds real
    /// content only for the Old summands (M < N).
    fn degeneracy_multiplicity(&self, summand: &HeckeSummand, m: u64) -> Result<u32, String> {
        let n = self.level();
        if m == n {
            return Ok(1);
        }
        let lower = ModularSymbolsGamma0::new(m);
        let pi = self.degeneracy_lowering_matrix(&lower, 1)?;
        let images: Vec<Vec<Rational>> = summand
            .ambient_basis()
            .iter()
            .map(|v| apply(&pi, v))
            .collect();
        let r = column_rank(&images, lower.dimension())?;
        if r == 0 {
            return Err(format!(
                "pi_1 annihilates a summand that the raising maps place inside Im({m}): \
                 the degeneracy maps are inconsistent"
            ));
        }
        let d = summand.dimension();
        if !d.is_multiple_of(r) {
            return Err(format!(
                "summand dimension {d} is not a multiple of rank(pi_1 | summand) = {r}, \
                 so it is not a whole number of degeneracy copies of the level-{m} block"
            ));
        }
        Ok((d / r) as u32)
    }

    /// Label a Hecke summand as New (of level N) or Old, with the source level
    /// M and the number of degeneracy copies it contains.
    ///
    /// The source level is the minimal M with the summand inside Im(M)
    /// ([`Self::summand_source_level`]); the multiplicity is then recomputed
    /// independently through the lowering maps and CHECKED against
    /// sigma_0(N/M), which is the free self-check the two disjoint routes buy
    /// us.  An `Err` here is a genuine finding, not a shrug: it says the
    /// summand is not a whole isotypic block of one newform orbit.
    ///
    /// The cross-check has teeth only for the OLD summands.  For a New summand
    /// (M == N) [`Self::degeneracy_multiplicity`] short-circuits to 1 and the
    /// comparison is 1 against sigma_0(1) = 1, which is vacuous; the New label is
    /// carried entirely by the minimality of the source level, i.e. by the
    /// raising-map scan in [`Self::summand_source_level`].
    ///
    /// Intended for the summands of [`Self::cuspidal_hecke_decomposition`].  A
    /// proper piece of one (e.g. a [`UlRefinement`] piece) still has a
    /// well-defined source level but obeys no sigma_0 law, and is correctly
    /// rejected here - use [`Self::summand_source_level`] for those.
    pub fn summand_provenance(&self, summand: &HeckeSummand) -> Result<SummandProvenance, String> {
        let n = self.level();
        let m = self.summand_source_level(summand)?;
        let multiplicity = self.degeneracy_multiplicity(summand, m)?;
        let expected = sigma0(n / m);
        if multiplicity != expected {
            return Err(format!(
                "summand at level {n} has source level {m} but contains {multiplicity} \
                 degeneracy copies, not sigma_0({}) = {expected}: it is not a whole \
                 isotypic block of a single level-{m} newform orbit",
                n / m
            ));
        }
        Ok(if m == n {
            SummandProvenance::New { level: n }
        } else {
            SummandProvenance::Old {
                source_level: m,
                multiplicity,
            }
        })
    }

    /// The action of U_l (l prime, l | N) on a summand: its exact
    /// characteristic polynomial, its certified factorization over Q, and the
    /// generalized eigenspaces as genuine finer Hecke summands.
    ///
    /// This is where [`SummandHeckeAction::Mixed`] turns from a deferral into a
    /// structure theorem.  Wherever charpoly(U_l) has several distinct
    /// irreducible factors, the refinement is PERFORMED and
    /// [`UlRefinement::pieces`] returns the finer summands (they are still
    /// T_p-stable, since U_l commutes with every T_p for p coprime to N).
    /// Wherever it does not, the block is kept whole and the irreducible factor
    /// is returned as the explicit obstruction ([`UlRefinement::obstructions`]).
    ///
    /// For l coprime to N this is refused: T_l is already irreducible on a
    /// summand by construction of the decomposition, so there is nothing here
    /// that [`Self::hecke_action_on_summand`] does not already say.
    ///
    /// See the module docs for exactly which of the three regimes (l not | M,
    /// l | M, l^2 | N/M) splits and which provably cannot.
    pub fn u_refinement_of_summand(
        &self,
        summand: &HeckeSummand,
        l: u64,
    ) -> Result<UlRefinement, String> {
        let n = self.level();
        if !is_prime_u64(l) {
            return Err(format!("U_l needs a prime; {l} is not one"));
        }
        if !n.is_multiple_of(l) {
            return Err(format!(
                "U_{l} needs l | N = {n}; for l coprime to the level T_l already acts with a \
                 single irreducible factor on every summand by construction - use \
                 hecke_action_on_summand"
            ));
        }
        if summand.dimension() == 0 {
            return Err("empty summand".to_string());
        }
        let u = self.hecke_matrix_cuspidal(l);
        let uw = restrict_to_column_span(&u, summand.cuspidal_coordinates())?;
        let charpoly = charpoly_berkowitz(&uw).expect("restricted Hecke matrix is square");
        let factors = factor_monic_rational_certified(&charpoly)?;
        let cusp_basis = self.cuspidal_basis();
        let mut pieces = Vec::with_capacity(factors.len());
        for (f, e) in &factors {
            let fdeg = f.degree().expect("irreducible factors have degree >= 1");
            // generalized eigenspace ker f(U_l)^e (primary decomposition, so no
            // semisimplicity of U_l is assumed - and U_l really is NOT
            // semisimple on the l^2 | N/M blocks, e.g. at level 44)
            let ker = mat_pow(&eval_poly_at_matrix(f, &uw), *e)
                .kernel()
                .map_err(|err| format!("kernel failed: {err:?}"))?;
            if ker.len() != fdeg * (*e as usize) {
                return Err(format!(
                    "generalized U_{l}-eigenspace of a degree-{fdeg} factor with multiplicity \
                     {e} has dimension {}, not {}",
                    ker.len(),
                    fdeg * (*e as usize)
                ));
            }
            let cuspidal_coords: Vec<Vec<Rational>> = ker
                .iter()
                .map(|x| lin_comb(summand.cuspidal_coordinates(), x))
                .collect();
            let ambient_basis = cuspidal_coords
                .iter()
                .map(|x| lin_comb(cusp_basis, x))
                .collect();
            pieces.push(UlPiece {
                factor: f.clone(),
                multiplicity: *e,
                summand: HeckeSummand::new(ambient_basis, cuspidal_coords),
            });
        }
        let total: usize = pieces.iter().map(|p| p.summand.dimension()).sum();
        if total != summand.dimension() {
            return Err(format!(
                "the U_{l} generalized eigenspaces have dimensions summing to {total}, not the \
                 summand's {}",
                summand.dimension()
            ));
        }
        Ok(UlRefinement {
            prime: l,
            charpoly,
            pieces,
        })
    }
}

/// Where a Hecke summand comes from: a newform of level N itself, or the
/// degeneracy images of a newform orbit of a proper divisor level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummandProvenance {
    /// The summand is NEW: it is killed by every degeneracy lowering map, and
    /// the minimal level whose raising images contain it is N itself.
    New {
        /// The level N (= the newform orbit's own level).
        level: u64,
    },
    /// The summand is OLD: it is the span of the sigma_0(N/M) degeneracy
    /// images delta_t (t | N/M) of one newform orbit of the strictly smaller
    /// level M.
    Old {
        /// The minimal level M | N, M < N whose degeneracy images contain the
        /// summand: the exact level of the underlying newform orbit.
        source_level: u64,
        /// The number of degeneracy copies, independently recomputed through
        /// the lowering maps and certified equal to sigma_0(N/M).
        multiplicity: u32,
    },
}

impl SummandProvenance {
    /// Whether the summand is new at level N.
    pub fn is_new(&self) -> bool {
        matches!(self, SummandProvenance::New { .. })
    }

    /// The level of the underlying newform orbit (N itself when new).
    pub fn source_level(&self) -> u64 {
        match *self {
            SummandProvenance::New { level } => level,
            SummandProvenance::Old { source_level, .. } => source_level,
        }
    }

    /// The number of degeneracy copies, = sigma_0(N / source_level), = 1 when new.
    pub fn multiplicity(&self) -> u32 {
        match *self {
            SummandProvenance::New { .. } => 1,
            SummandProvenance::Old { multiplicity, .. } => multiplicity,
        }
    }
}

/// One generalized U_l-eigenspace of a summand: an irreducible factor of
/// charpoly(U_l), its multiplicity, and the corresponding finer Hecke summand.
#[derive(Debug, Clone)]
pub struct UlPiece {
    factor: UnivariatePolynomial<Rational>,
    multiplicity: u32,
    summand: HeckeSummand,
}

impl UlPiece {
    /// The monic irreducible factor of charpoly(U_l) cutting this piece out.
    /// Degree 1 means U_l has a rational eigenvalue here; degree > 1 means this
    /// piece cannot be split further by U_l over Q, and the factor IS the
    /// obstruction.
    pub fn factor(&self) -> &UnivariatePolynomial<Rational> {
        &self.factor
    }

    /// The multiplicity of the factor in charpoly(U_l).
    pub fn multiplicity(&self) -> u32 {
        self.multiplicity
    }

    /// The piece itself: ker f(U_l)^e, a Hecke-stable summand (U_l commutes
    /// with every T_p for p coprime to N, so the T_p eigensystem is unchanged).
    pub fn summand(&self) -> &HeckeSummand {
        &self.summand
    }
}

/// The action of U_l on a summand, decomposed as far as Q allows.
#[derive(Debug, Clone)]
pub struct UlRefinement {
    prime: u64,
    charpoly: UnivariatePolynomial<Rational>,
    pieces: Vec<UlPiece>,
}

impl UlRefinement {
    /// The prime l (dividing the level).
    pub fn prime(&self) -> u64 {
        self.prime
    }

    /// The exact characteristic polynomial of U_l on the summand.
    pub fn charpoly(&self) -> &UnivariatePolynomial<Rational> {
        &self.charpoly
    }

    /// The generalized eigenspaces, one per distinct irreducible factor.
    pub fn pieces(&self) -> &[UlPiece] {
        &self.pieces
    }

    /// Whether U_l actually refines the summand over Q, i.e. charpoly(U_l) has
    /// at least two distinct irreducible factors.  False means the block stays
    /// whole and [`Self::obstructions`] says why.
    pub fn splits(&self) -> bool {
        self.pieces.len() > 1
    }

    /// The irreducible factors of degree > 1: the exact obstructions to
    /// splitting the summand further with U_l over Q.  Empty iff U_l has only
    /// rational eigenvalues on the summand.
    ///
    /// For an old block of a newform of level M and a prime l | N/M with
    /// l not dividing M, this always contains x^2 - a_l x + l, which is
    /// irreducible over Q for EVERY newform (module docs: a rational root would
    /// force |a_l| > 2 sqrt(l), contradicting Ramanujan-Petersson).  That is
    /// the precise sense in which the sigma_0(N/M) degeneracy copies cannot be
    /// separated over Q.
    pub fn obstructions(&self) -> Vec<&UnivariatePolynomial<Rational>> {
        self.pieces
            .iter()
            .filter(|p| p.factor.degree().is_some_and(|d| d > 1))
            .map(|p| &p.factor)
            .collect()
    }

    /// Whether U_l has only rational eigenvalues on the summand (every
    /// irreducible factor linear), so no obstruction remains.
    pub fn splits_completely(&self) -> bool {
        self.obstructions().is_empty()
    }
}

impl CuspidalHeckeDecomposition {
    /// Label one summand New / Old(M, multiplicity).  Rebuilds the ambient
    /// space from the recorded level; prefer [`Self::provenances`] when
    /// labelling all of them.
    pub fn summand_provenance(&self, summand: &HeckeSummand) -> Result<SummandProvenance, String> {
        ModularSymbolsGamma0::new(self.level()).summand_provenance(summand)
    }

    /// Label every summand, in the order of [`Self::summands`], building the
    /// ambient space once.
    pub fn provenances(&self) -> Result<Vec<SummandProvenance>, String> {
        let space = ModularSymbolsGamma0::new(self.level());
        self.summands()
            .iter()
            .map(|w| space.summand_provenance(w))
            .collect()
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

    /// dim S_2^new(Gamma0(m)) via the Mobius inversion in `dims`.
    fn new_dim(m: u64) -> usize {
        use rustmath_core::NumericConversion;
        crate::dims::dimension_new_cusp_forms(&Integer::from(m), 2)
            .to_u64()
            .expect("a new dimension is a small nonnegative integer") as usize
    }

    /// THE SELF-CERTIFYING GATE.  For every level N up to the cap, the
    /// weight-2 multiplicity-one dimension law is verified THREE independent
    /// ways and cross-asserted (a modular symbol space has dimension 2 * the
    /// dimension of the corresponding space of forms, so every count below is
    /// doubled):
    ///
    ///   (1) dim S-part of the actual Manin symbol space (= 2 g(X_0(N)));
    ///   (2) sum_{M | N} sigma_0(N/M) * 2 dim S_2^new(M), from the fixed
    ///       `dims::dimension_new_cusp_forms` (Mobius inversion with
    ///       beta = mu * mu, on top of the exact genus formula);
    ///   (3) dim cuspidal_new_subspace() + dim cuspidal_old_subspace(), from
    ///       the actual kernels of the lowering maps and the actual spans of
    ///       the raising images.
    ///
    /// Plus: the new subspace has dimension exactly 2 dim S_2^new(N), and new
    /// and old TOGETHER SPAN the cuspidal space (so the dimension count is not
    /// an accident of two errors cancelling).  A bug in the degeneracy maps,
    /// in the Mobius inversion, or in the genus formula each breaks this
    /// independently.
    ///
    /// The cap is N = 100: the whole test runs in a few seconds, and 100 is
    /// also the range of the crate's GENUS_X0 / CUSPS_X0 reference tables.
    #[test]
    fn test_dimension_law_three_ways() {
        for n in 1..=100u64 {
            let m = ModularSymbolsGamma0::new(n);
            let cuspidal = m.cuspidal_dimension();

            // (2) the dimension law from dims.rs
            let predicted: usize = divisors(n)
                .into_iter()
                .map(|d| sigma0(n / d) as usize * 2 * new_dim(d))
                .sum();
            assert_eq!(
                cuspidal, predicted,
                "dim S-part of the Manin space at {n} vs sum_M sigma_0(N/M) dim new(M)"
            );

            // (3) the actual subspaces
            let new = m.cuspidal_new_subspace().unwrap();
            let old = m.cuspidal_old_subspace().unwrap();
            assert_eq!(
                new.len() + old.len(),
                cuspidal,
                "dim(new) + dim(old) != dim(cuspidal) at level {n}"
            );
            assert_eq!(
                new.len(),
                2 * new_dim(n),
                "dim of the computed new subspace at {n} vs dims.rs"
            );
            // not just the counts: new and old together SPAN the cuspidal
            // space (both are given in cuspidal-basis coordinates)
            let mut cols = new;
            cols.extend(old);
            assert_eq!(
                rank(&cols, cuspidal),
                cuspidal,
                "new + old must span the cuspidal space at level {n}"
            );
        }
    }

    /// GATE: every vector of the old subspace really is cuspidal, really is a
    /// combination of degeneracy raising images, and the old subspace is
    /// Hecke stable for primes coprime to the level (T_q old = old).
    #[test]
    fn test_old_subspace_is_cuspidal_and_hecke_stable() {
        for (n, qs) in [(22u64, vec![3u64, 5]), (33, vec![2, 5]), (44, vec![3, 5]), (45, vec![2, 7])]
        {
            let m = ModularSymbolsGamma0::new(n);
            let old = m.cuspidal_old_subspace().unwrap();
            assert!(!old.is_empty(), "level {n} has a nonzero old space");
            let cb = m.cuspidal_basis();
            let ambient: Vec<Vec<Rational>> = old.iter().map(|x| lin_comb(cb, x)).collect();
            for v in &ambient {
                assert!(m.is_cuspidal(v), "old vector at {n} must be cuspidal");
            }
            let d = rank(&ambient, m.dimension());
            for q in qs {
                let tq = m.hecke_matrix(q);
                let mut cols = ambient.clone();
                for v in &ambient {
                    cols.push(mv(&tq, v));
                }
                assert_eq!(
                    rank(&cols, m.dimension()),
                    d,
                    "T_{q} must preserve the old subspace at level {n}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // STAGE 2: old/new LABELLING of the summands, and the U_l action
    //
    // Every elliptic-curve eigenvalue asserted below is POINT-COUNTED from an
    // integral Weierstrass model by the three helpers that follow, and the
    // model's conductor is re-derived from (Delta, c4) rather than trusted:
    // no table is consulted anywhere in this section.
    // ---------------------------------------------------------------------

    /// Minimal Weierstrass models [a1, a2, a3, a4, a6].  Only the MODEL is an
    /// input; the conductor and every a_p are recomputed from it below.
    const E11A: [i64; 5] = [0, -1, 1, -10, -20];
    const E14A: [i64; 5] = [1, 0, 1, 4, -6];
    const E15A: [i64; 5] = [1, 1, 1, -10, -10];
    const E21A: [i64; 5] = [1, 0, 0, -4, -1];
    const E33A: [i64; 5] = [1, 1, 0, -11, 0];
    const E42A: [i64; 5] = [1, 1, 1, -4, 5];

    /// (Delta, c4) of an integral Weierstrass model.
    fn delta_c4(m: [i64; 5]) -> (i64, i64) {
        let (a1, a2, a3, a4, a6) = (m[0], m[1], m[2], m[3], m[4]);
        let b2 = a1 * a1 + 4 * a2;
        let b4 = 2 * a4 + a1 * a3;
        let b6 = a3 * a3 + 4 * a6;
        let b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4;
        let disc = -b2 * b2 * b8 - 8 * b4 * b4 * b4 - 27 * b6 * b6 + 9 * b2 * b4 * b6;
        (disc, b2 * b2 - 24 * b4)
    }

    /// The conductor of a SEMISTABLE curve, re-derived from the model: E is
    /// semistable iff no prime dividing Delta divides c4 (multiplicative
    /// reduction everywhere), and then N = rad(Delta).  Panics if the
    /// criterion does not apply, so a wrong model cannot slip through as a
    /// plausible-looking number.
    fn conductor_of(m: [i64; 5]) -> u64 {
        let (disc, c4) = delta_c4(m);
        assert!(disc != 0, "singular model");
        let mut d = disc.unsigned_abs();
        let mut rad = 1u64;
        let mut p = 2u64;
        while p * p <= d {
            if d.is_multiple_of(p) {
                assert!(
                    c4.rem_euclid(p as i64) != 0,
                    "p = {p} divides both Delta and c4: the curve is NOT semistable, so \
                     conductor = rad(Delta) does not apply"
                );
                rad *= p;
                while d.is_multiple_of(p) {
                    d /= p;
                }
            }
            p += 1;
        }
        if d > 1 {
            assert!(
                c4.rem_euclid(d as i64) != 0,
                "the curve is not semistable at {d}"
            );
            rad *= d;
        }
        rad
    }

    /// a_p = p + 1 - #E(F_p) at a GOOD prime, by brute-force point counting
    /// over F_p (Eichler-Shimura: this is the T_p eigenvalue of the newform).
    /// At a BAD (necessarily multiplicative, by `conductor_of`) prime the same
    /// count is taken over the NONSINGULAR points and a_l = l - #E_ns(F_l),
    /// which is the U_l eigenvalue.
    fn ap_point_count(m: [i64; 5], p: u64) -> i64 {
        let (a1, a2, a3, a4, a6) = (m[0], m[1], m[2], m[3], m[4]);
        let (disc, _) = delta_c4(m);
        let pi = p as i64;
        let good = disc.rem_euclid(pi) != 0;
        let mut count = 1i64; // the point at infinity (always nonsingular)
        for x in 0..pi {
            for y in 0..pi {
                let f = (y * y + a1 * x * y + a3 * y
                    - (x * x * x + a2 * x * x + a4 * x + a6))
                    .rem_euclid(pi);
                if f != 0 {
                    continue;
                }
                if good {
                    count += 1;
                } else {
                    // keep only nonsingular points: some partial is nonzero
                    let dx = (-(3 * x * x + 2 * a2 * x + a4) + a1 * y).rem_euclid(pi);
                    let dy = (2 * y + a1 * x + a3).rem_euclid(pi);
                    if dx != 0 || dy != 0 {
                        count += 1;
                    }
                }
            }
        }
        if good {
            pi + 1 - count
        } else {
            pi - count
        }
    }

    /// The rational eigenvalue of T_n on a summand (panics if not rational).
    fn rat_eig(m: &ModularSymbolsGamma0, w: &HeckeSummand, n: u64) -> i64 {
        use crate::modsym::decomposition::{HeckeEigenvalue, SummandHeckeAction};
        match m.hecke_action_on_summand(w, n).unwrap() {
            SummandHeckeAction::Eigenvalue(HeckeEigenvalue::Rational(a)) => {
                assert!(a.denominator().is_one(), "integral eigenvalue");
                a.numerator().to_i64()
            }
            other => panic!("expected a rational T_{n} eigenvalue, got {other:?}"),
        }
    }

    /// Find the unique summand with the given provenance.
    fn summand_with<'a>(
        m: &ModularSymbolsGamma0,
        dec: &'a CuspidalHeckeDecomposition,
        want: SummandProvenance,
    ) -> &'a HeckeSummand {
        let hits: Vec<&HeckeSummand> = dec
            .summands()
            .iter()
            .filter(|w| m.summand_provenance(w) == Ok(want))
            .collect();
        assert_eq!(hits.len(), 1, "exactly one summand with provenance {want:?}");
        hits[0]
    }

    /// The point-counting helpers really do reproduce the curves they claim,
    /// with the conductor DERIVED from (Delta, c4), not asserted from a table.
    /// Ramanujan-Hasse |a_p| <= 2 sqrt(p) is checked on every count, and the
    /// bad primes give |a_l| = 1 (multiplicative reduction) as they must.
    #[test]
    fn test_point_counting_helpers_are_sound() {
        for (m, n) in [
            (E11A, 11u64),
            (E14A, 14),
            (E15A, 15),
            (E21A, 21),
            (E33A, 33),
            (E42A, 42),
        ] {
            assert_eq!(conductor_of(m), n, "conductor from Delta/c4");
            for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23] {
                let a = ap_point_count(m, p);
                if n.is_multiple_of(p) {
                    assert_eq!(a.abs(), 1, "multiplicative a_{p} = +-1 for conductor {n}");
                } else {
                    assert!(a * a <= 4 * p as i64, "Hasse bound for conductor {n} at {p}");
                }
            }
        }
        // the specific values the gates below lean on
        assert_eq!(ap_point_count(E11A, 2), -2);
        assert_eq!(ap_point_count(E14A, 3), -2);
        assert_eq!(ap_point_count(E21A, 2), -1);
        assert_eq!(ap_point_count(E33A, 2), 1);
    }

    /// GATE: delta_1 from a level to ITSELF is the identity.  This is what
    /// lets the uniform divisor scan of `summand_source_level` label a NEW
    /// summand (source level N) by exactly the same code path as an old one:
    /// Im(N) is the whole cuspidal space, with no special case.
    #[test]
    fn test_delta_1_from_own_level_is_identity() {
        for n in [11u64, 22, 33, 42] {
            let m = ModularSymbolsGamma0::new(n);
            let d = m.degeneracy_raising_matrix(&m, 1).unwrap();
            assert_eq!(d, Matrix::identity(m.dimension()), "delta_1^({n}<-{n}) = id");
            let img = m.degeneracy_image_from_level(n).unwrap();
            assert_eq!(
                img.len(),
                m.cuspidal_dimension(),
                "Im({n}) is the whole cuspidal space"
            );
        }
    }

    /// GATE: the provenance labels of the levels the crate already tested,
    /// now with the source level and the sigma_0(N/M) multiplicity attached.
    /// `summand_is_new` (the stage-1 API) must agree with the label.
    #[test]
    fn test_summand_provenance_labels_known_levels() {
        let expect: &[(u64, &[SummandProvenance])] = &[
            (11, &[SummandProvenance::New { level: 11 }]),
            (23, &[SummandProvenance::New { level: 23 }]),
            (
                22,
                &[SummandProvenance::Old {
                    source_level: 11,
                    multiplicity: 2,
                }],
            ),
            (
                33,
                &[
                    SummandProvenance::New { level: 33 },
                    SummandProvenance::Old {
                        source_level: 11,
                        multiplicity: 2,
                    },
                ],
            ),
            (
                44,
                &[
                    SummandProvenance::New { level: 44 },
                    SummandProvenance::Old {
                        source_level: 11,
                        multiplicity: 3,
                    },
                ],
            ),
            (
                45,
                &[
                    SummandProvenance::New { level: 45 },
                    SummandProvenance::Old {
                        source_level: 15,
                        multiplicity: 2,
                    },
                ],
            ),
        ];
        for (n, want) in expect {
            let m = ModularSymbolsGamma0::new(*n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            assert_eq!(&dec.provenances().unwrap(), want, "provenances at level {n}");
            // the stage-1 boolean API must agree with the richer label
            for (w, p) in dec.summands().iter().zip(want.iter()) {
                assert_eq!(m.summand_is_new(w), Ok(p.is_new()), "summand_is_new at {n}");
                assert_eq!(
                    p.multiplicity(),
                    sigma0(n / p.source_level()),
                    "multiplicity = sigma_0(N/M) at {n}"
                );
            }
        }
    }

    /// GATE (N = 42 = 2 * 3 * 7, genus 5): the smallest level whose OLD part
    /// contains TWO DISTINCT newform orbits - 14a with sigma_0(42/14) = 2
    /// degeneracy images and 21a with sigma_0(42/21) = 2 - alongside the
    /// 2-dimensional new part 42a.  Dimension law: 2*(2*1 + 2*1 + 1) = 10.
    ///
    /// Being blunt about what is and is not new here: the T_p refinement
    /// ALREADY separates 14a from 21a for free, because they are different
    /// eigensystems (a_5 = 0 vs -2).  The new content of this gate is the
    /// LABELLING - that the two 4-dimensional blocks are correctly identified
    /// as coming from levels 14 and 21 respectively, each with multiplicity 2 -
    /// and the Eichler-Shimura identification of their eigenvalues with
    /// point-counted a_p of the two elliptic curves.
    #[test]
    fn test_level_42_two_old_orbits_labelled_and_new_part() {
        let m = ModularSymbolsGamma0::new(42);
        assert_eq!(m.cuspidal_dimension(), 10, "2 g(X0(42)) = 10");
        let dec = m.cuspidal_hecke_decomposition().unwrap();

        // three-way dimension law at 42 (the batteries stop at 40)
        let new = m.cuspidal_new_subspace().unwrap();
        let old = m.cuspidal_old_subspace().unwrap();
        assert_eq!(new.len(), 2, "dim new = 2 * dim S_2^new(42) = 2");
        assert_eq!(old.len(), 8, "dim old = 2 * (2 * 1 + 2 * 1) = 8");
        assert_eq!(new.len() + old.len(), 10);
        let mut cols = new;
        cols.extend(old);
        assert_eq!(rank(&cols, 10), 10, "new + old span the cuspidal space at 42");

        // the labels
        assert_eq!(
            dec.provenances().unwrap(),
            vec![
                SummandProvenance::New { level: 42 },
                SummandProvenance::Old {
                    source_level: 14,
                    multiplicity: 2
                },
                SummandProvenance::Old {
                    source_level: 21,
                    multiplicity: 2
                },
            ],
            "42 = New(42) + Old(14, x2) + Old(21, x2)"
        );

        // EICHLER-SHIMURA: the T_p eigenvalues of each block are the
        // POINT-COUNTED a_p of the corresponding curve (conductor re-derived
        // from Delta/c4 inside conductor_of, so nothing is table-fed)
        for (curve, source, mult) in [(E14A, 14u64, 2u32), (E21A, 21, 2), (E42A, 42, 1)] {
            assert_eq!(conductor_of(curve), source);
            let want = if source == 42 {
                SummandProvenance::New { level: 42 }
            } else {
                SummandProvenance::Old {
                    source_level: source,
                    multiplicity: mult,
                }
            };
            let w = summand_with(&m, &dec, want);
            assert_eq!(w.dimension() as u32, 2 * mult, "dim = 2 * sigma_0(42/{source})");
            for p in [5u64, 11, 13, 17, 19, 23, 29] {
                assert_eq!(
                    rat_eig(&m, w, p),
                    ap_point_count(curve, p),
                    "T_{p} on the level-{source} block = point-counted a_{p}"
                );
            }
            // U_l for l | 42 that ALSO divides the source level acts as the
            // scalar a_l (point-counted at the multiplicative prime)
            for l in [2u64, 3, 7] {
                if source.is_multiple_of(l) {
                    assert_eq!(
                        rat_eig(&m, w, l),
                        ap_point_count(curve, l),
                        "U_{l} = a_{l} (scalar) on the level-{source} block"
                    );
                }
            }
        }
    }

    /// GATE (N = 66 = 2 * 3 * 11, genus 9): the smallest NESTED old tower.
    /// 11a is old at 66 with sigma_0(6) = 4 images, and 33a is old at 66 with
    /// sigma_0(2) = 2 images - but 11 | 33 | 66, so the level-11 forms are
    /// ALSO old at 33 and at 22.  The 11a block therefore lies inside the
    /// degeneracy image from EVERY ONE of the levels 11, 22, 33 and 66, and
    /// only the MINIMALITY of the source-level scan makes the answer 11.
    /// This test asserts that containment explicitly, so a "first M that
    /// works" implementation scanning divisors in any other order, or a
    /// "largest M", or a "maximal proper divisor" implementation, all fail it.
    ///
    /// dim law: 2*(sigma_0(6)*1 + sigma_0(2)*1 + 3) = 2*(4 + 2 + 3) = 18.
    /// Note S_2^new(22) = 0, so there is deliberately NO Old(22) block even
    /// though S_2(Gamma0(22)) itself is 2-dimensional: Im(22) is nonzero and
    /// contains the whole 11a block, which is exactly the trap.
    #[test]
    fn test_level_66_nested_old_tower_takes_the_minimal_source_level() {
        let m = ModularSymbolsGamma0::new(66);
        assert_eq!(m.cuspidal_dimension(), 18, "2 g(X0(66)) = 18");
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        assert_eq!(
            dec.provenances().unwrap(),
            vec![
                SummandProvenance::New { level: 66 },
                SummandProvenance::New { level: 66 },
                SummandProvenance::New { level: 66 },
                SummandProvenance::Old {
                    source_level: 33,
                    multiplicity: 2
                },
                SummandProvenance::Old {
                    source_level: 11,
                    multiplicity: 4
                },
            ],
            "66 = three rational newforms + Old(33, x2) + Old(11, x4)"
        );

        let w11 = summand_with(
            &m,
            &dec,
            SummandProvenance::Old {
                source_level: 11,
                multiplicity: 4,
            },
        );
        assert_eq!(w11.dimension(), 8, "sigma_0(6) = 4 images of a 2-dim block");

        // THE TRAP, ASSERTED: the 11a block sits inside the degeneracy image
        // from 22 and from 33 as well - and Im(22) is genuinely nonzero even
        // though level 22 has NO newforms of its own.
        let dim = m.dimension();
        for src in [11u64, 22, 33, 66] {
            let img = m.degeneracy_image_from_level(src).unwrap();
            assert!(!img.is_empty(), "Im({src}) is nonzero");
            assert!(
                span_contains(&img, w11.ambient_basis(), dim).unwrap(),
                "the 11a block lies inside Im({src}) - so only MINIMALITY picks 11"
            );
        }
        assert_eq!(
            m.degeneracy_image_from_level(22).unwrap().len(),
            8,
            "Im(22) is 8-dimensional and consists entirely of level-11 forms"
        );
        // and the 33a block is NOT in Im(11) or Im(22): its source really is 33
        let w33 = summand_with(
            &m,
            &dec,
            SummandProvenance::Old {
                source_level: 33,
                multiplicity: 2,
            },
        );
        for src in [11u64, 22] {
            let img = m.degeneracy_image_from_level(src).unwrap();
            assert!(
                !span_contains(&img, w33.ambient_basis(), dim).unwrap(),
                "the 33a block must NOT lie inside Im({src})"
            );
        }

        // Eichler-Shimura on both old blocks, against point-counted a_p
        for (curve, source, mult) in [(E11A, 11u64, 4u32), (E33A, 33, 2)] {
            assert_eq!(conductor_of(curve), source);
            let w = summand_with(
                &m,
                &dec,
                SummandProvenance::Old {
                    source_level: source,
                    multiplicity: mult,
                },
            );
            for p in [5u64, 7, 13, 17, 19, 23] {
                assert_eq!(
                    rat_eig(&m, w, p),
                    ap_point_count(curve, p),
                    "T_{p} on the level-{source} block at 66"
                );
            }
            // U_11 = a_11 (scalar) on both, since 11 divides both 11 and 33
            assert_eq!(rat_eig(&m, w, 11), ap_point_count(curve, 11));
        }
        // the three new summands carry three DISTINCT eigensystems
        let mut systems: Vec<Vec<i64>> = dec
            .summands()
            .iter()
            .filter(|w| m.summand_provenance(w) == Ok(SummandProvenance::New { level: 66 }))
            .map(|w| [5u64, 7, 13, 17].iter().map(|&p| rat_eig(&m, w, p)).collect())
            .collect();
        systems.sort();
        systems.dedup();
        assert_eq!(systems.len(), 3, "three distinct rational newforms at 66");
    }

    /// GATE - THE OBSTRUCTION, AND ITS PROOF.  For a newform f of level M and
    /// a prime l | N/M with l NOT dividing M, U_l on the plane spanned by
    /// delta_1(f), delta_l(f) has charpoly x^2 - a_l x + l, and this is
    /// IRREDUCIBLE over Q for EVERY newform: a rational root lambda would give
    /// a_l = lambda + l/lambda with |a_l| >= 2 sqrt(l) by AM-GM (equality only
    /// at the irrational lambda = +-sqrt(l)), contradicting Ramanujan-Petersson
    /// |a_l| <= 2 sqrt(l).  So the sigma_0(N/M) degeneracy copies of f can
    /// NEVER be separated over Q by U_l.  This is a theorem, not a gap in the
    /// implementation.
    ///
    /// Here the whole chain is re-run from scratch on four old blocks: the
    /// point-counted a_l of the source curve predicts the obstruction
    /// x^2 - a_l x + l, that polynomial IS what `u_refinement_of_summand`
    /// returns, its discriminant a_l^2 - 4l is negative, and the block is
    /// reported as NOT splitting.
    #[test]
    fn test_u_l_obstruction_is_irreducible_over_q() {
        // (N, curve, source level M, prime l | N/M with l not | M)
        for (n, curve, source, l) in [
            (22u64, E11A, 11u64, 2u64),
            (42, E14A, 14, 3),
            (42, E21A, 21, 2),
            (66, E11A, 11, 2),
            (66, E11A, 11, 3),
            (66, E33A, 33, 2),
        ] {
            assert_eq!(conductor_of(curve), source);
            assert!(!source.is_multiple_of(l) && (n / source).is_multiple_of(l));
            let m = ModularSymbolsGamma0::new(n);
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            let mult = sigma0(n / source);
            let w = summand_with(
                &m,
                &dec,
                SummandProvenance::Old {
                    source_level: source,
                    multiplicity: mult,
                },
            );
            let a_l = ap_point_count(curve, l);
            // Ramanujan-Hasse, strict because l is not a perfect square
            assert!(
                a_l * a_l - 4 * (l as i64) < 0,
                "disc = a_l^2 - 4l must be NEGATIVE: a_{l} = {a_l}, l = {l}"
            );
            let expected = poly(&[l as i64, -a_l, 1]); // x^2 - a_l x + l
            let r = m.u_refinement_of_summand(w, l).unwrap();
            assert_eq!(r.prime(), l);
            assert!(
                !r.splits(),
                "U_{l} must NOT split the level-{source} block at {n} (irreducible charpoly factor)"
            );
            assert_eq!(
                r.obstructions(),
                vec![&expected],
                "the obstruction is exactly x^2 - a_{l} x + {l} with the point-counted a_{l} = {a_l}"
            );
            assert!(!r.splits_completely());
            assert_eq!(r.pieces().len(), 1);
            assert_eq!(r.pieces()[0].summand().dimension(), w.dimension());
            // charpoly = (x^2 - a_l x + l)^(dim/2)
            let mut cp = poly(&[1]);
            for _ in 0..(w.dimension() / 2) {
                cp = cp * expected.clone();
            }
            assert_eq!(r.charpoly(), &cp, "charpoly(U_{l}) on the {n}-old block");
        }
    }

    /// GATE - WHERE U_l DOES SPLIT, THE REFINEMENT IS PERFORMED.  Two regimes,
    /// both already present in the crate's fixtures and now resolved into
    /// actual finer summands:
    ///
    ///  * N = 45, M = 15, l = 3 with 3 | M: charpoly x(x - a_3), a_3 = -1
    ///    point-counted, two DISTINCT rational roots, so the block splits
    ///    COMPLETELY over Q (no obstruction left).
    ///  * N = 44, M = 11, l = 2 with l^2 | N/M = 4: charpoly
    ///    x^2 (x^2 - a_2 x + 2)^2 - the nilpotent part splits off over Q, but
    ///    the quadratic factor survives, so the split is PARTIAL and the
    ///    obstruction is still reported.
    ///
    /// In both cases the pieces are genuine Hecke summands, they still have
    /// source level M, and their dimensions sum back to the block's.
    #[test]
    fn test_u_refinement_splits_where_q_allows() {
        // ---- 45: l | M, splits completely ----
        let m45 = ModularSymbolsGamma0::new(45);
        let dec45 = m45.cuspidal_hecke_decomposition().unwrap();
        let w15 = summand_with(
            &m45,
            &dec45,
            SummandProvenance::Old {
                source_level: 15,
                multiplicity: 2,
            },
        );
        let a3 = ap_point_count(E15A, 3);
        assert_eq!(a3, -1, "point-counted a_3(15a) at the multiplicative prime 3");
        let r = m45.u_refinement_of_summand(w15, 3).unwrap();
        assert!(r.splits(), "U_3 splits the 15-old block at 45");
        assert!(
            r.splits_completely(),
            "l | M: charpoly x(x - a_3) has two rational roots, no obstruction"
        );
        assert!(r.obstructions().is_empty());
        assert_eq!(
            r.charpoly(),
            &(poly(&[0, 1]) * poly(&[0, 1]) * poly(&[-a3, 1]) * poly(&[-a3, 1])),
            "charpoly(U_3) = x^2 (x - a_3)^2"
        );
        let mut got: Vec<(Vec<i64>, u32, usize)> = r
            .pieces()
            .iter()
            .map(|p| {
                (
                    p.factor()
                        .coefficients()
                        .iter()
                        .map(|c| c.numerator().to_i64())
                        .collect(),
                    p.multiplicity(),
                    p.summand().dimension(),
                )
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![(vec![0, 1], 2, 2), (vec![1, 1], 2, 2)],
            "U_3 pieces: eigenvalue 0 and eigenvalue a_3 = -1, each 2-dimensional"
        );
        // the pieces are still level-15-old, and still carry the 15a T_p system
        for p in r.pieces() {
            assert_eq!(m45.summand_source_level(p.summand()), Ok(15));
            for q in [2u64, 7, 11, 13] {
                assert_eq!(
                    rat_eig(&m45, p.summand(), q),
                    ap_point_count(E15A, q),
                    "the U_3 pieces keep the 15a eigensystem"
                );
            }
        }

        // ---- 44: l^2 | N/M, partial split ----
        let m44 = ModularSymbolsGamma0::new(44);
        let dec44 = m44.cuspidal_hecke_decomposition().unwrap();
        let w11 = summand_with(
            &m44,
            &dec44,
            SummandProvenance::Old {
                source_level: 11,
                multiplicity: 3,
            },
        );
        let a2 = ap_point_count(E11A, 2);
        assert_eq!(a2, -2);
        let r = m44.u_refinement_of_summand(w11, 2).unwrap();
        assert!(r.splits(), "U_2 DOES refine the 11-old block at 44");
        assert!(
            !r.splits_completely(),
            "but only partially: the quadratic factor survives"
        );
        let quad = poly(&[2, -a2, 1]); // x^2 + 2x + 2
        assert_eq!(r.obstructions(), vec![&quad]);
        assert_eq!(
            r.charpoly(),
            &(poly(&[0, 1]) * poly(&[0, 1]) * quad.clone() * quad.clone()),
            "charpoly(U_2) = x^2 (x^2 - a_2 x + 2)^2"
        );
        let mut dims: Vec<(usize, usize)> = r
            .pieces()
            .iter()
            .map(|p| (p.factor().degree().unwrap(), p.summand().dimension()))
            .collect();
        dims.sort();
        assert_eq!(
            dims,
            vec![(1, 2), (2, 4)],
            "the nilpotent line (dim 2) splits off; the irreducible part (dim 4) does not"
        );
        for p in r.pieces() {
            assert_eq!(m44.summand_source_level(p.summand()), Ok(11));
        }
    }

    /// GATE: the multiplicity self-check inside `summand_provenance` HAS TEETH.
    /// A U_l-refined piece is a perfectly good Hecke-stable subspace with a
    /// well-defined source level, but it is NOT a whole isotypic block, so it
    /// carries the wrong number of degeneracy copies - and provenance must
    /// REFUSE it rather than invent a label.  (This is the same check that
    /// would fire if the raising and lowering maps ever disagreed.)
    #[test]
    fn test_provenance_multiplicity_check_rejects_a_partial_block() {
        let m = ModularSymbolsGamma0::new(45);
        let dec = m.cuspidal_hecke_decomposition().unwrap();
        let w15 = summand_with(
            &m,
            &dec,
            SummandProvenance::Old {
                source_level: 15,
                multiplicity: 2,
            },
        );
        let r = m.u_refinement_of_summand(w15, 3).unwrap();
        assert_eq!(r.pieces().len(), 2);
        for p in r.pieces() {
            // the source level is still well defined ...
            assert_eq!(m.summand_source_level(p.summand()), Ok(15));
            // ... but the sigma_0 law does not hold on half a block
            let err = m
                .summand_provenance(p.summand())
                .expect_err("a half-block must not be labelled");
            assert!(
                err.contains("degeneracy copies") || err.contains("pi_1 annihilates"),
                "the refusal must name the reason, got: {err}"
            );
        }
    }

    /// GATE: over every level with a nonzero cuspidal space up to 50, EVERY
    /// summand gets a label, and the labels reproduce the Mobius-inverted
    /// dimension formula ORBIT BY ORBIT, not just in total:
    ///   for each source level M, the summands labelled Old(M)/New(M) have
    ///   dimensions summing to sigma_0(N/M) * 2 * dim S_2^new(M).
    /// A wrong source level (e.g. 22 or 33 instead of 11 at level 66) moves
    /// dimension into a bucket whose `dimension_new_cusp_forms` is zero and
    /// breaks this immediately.
    #[test]
    fn test_provenance_battery_reproduces_the_dimension_formula_per_orbit() {
        for n in 11..=50u64 {
            let m = ModularSymbolsGamma0::new(n);
            if m.cuspidal_dimension() == 0 {
                continue;
            }
            let dec = m.cuspidal_hecke_decomposition().unwrap();
            let provs = dec
                .provenances()
                .unwrap_or_else(|e| panic!("labelling failed at level {n}: {e}"));
            // bucket the summand dimensions by source level
            let mut by_source: Vec<(u64, usize)> = Vec::new();
            for (w, p) in dec.summands().iter().zip(provs.iter()) {
                assert_eq!(
                    m.summand_is_new(w),
                    Ok(p.is_new()),
                    "summand_is_new disagrees with the label at level {n}"
                );
                assert_eq!(
                    p.multiplicity(),
                    sigma0(n / p.source_level()),
                    "multiplicity != sigma_0(N/M) at level {n}"
                );
                let m0 = p.source_level();
                match by_source.iter_mut().find(|(s, _)| *s == m0) {
                    Some((_, d)) => *d += w.dimension(),
                    None => by_source.push((m0, w.dimension())),
                }
            }
            for m0 in divisors(n) {
                let got = by_source
                    .iter()
                    .find(|(s, _)| *s == m0)
                    .map_or(0, |(_, d)| *d);
                let want = sigma0(n / m0) as usize * 2 * new_dim(m0);
                assert_eq!(
                    got, want,
                    "level {n}: summands labelled with source {m0} have total dimension {got}, \
                     but sigma_0({}) * 2 * dim S_2^new({m0}) = {want}",
                    n / m0
                );
            }
        }
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
