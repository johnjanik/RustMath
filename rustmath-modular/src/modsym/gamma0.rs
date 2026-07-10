//! # Modular symbols for Gamma0(N), weight 2, trivial character, over Q
//!
//! The Manin-symbol presentation of the space M_2(Gamma0(N)) of weight-2
//! modular symbols: generators are the points (c : d) of P^1(Z/NZ), subject
//! to the relations x + xS = 0 and x + xT + xT^2 = 0 where
//! S = [[0,-1],[1,0]] and T = [[0,-1],[1,-1]] act on the right.  The
//! quotient is computed exactly over Q by row reduction, giving an explicit
//! basis, a projection map from generators to basis coordinates, the
//! boundary map to the cusps (whose kernel is the cuspidal subspace), and
//! conversion of arbitrary modular symbols {alpha, beta} to basis
//! coordinates via continued fractions (the Manin trick).
//!
//! Corresponds to `sage.modular.modsym` (ambient, boundary, manin_symbol)
//! and the MAGMA handbook chapter "Modular Symbols".  The algorithms follow
//! Cremona, *Algorithms for Modular Elliptic Curves*, ch. 2, and Stein,
//! *Modular Forms: A Computational Approach*, ch. 3 and 8.
//!
//! Verified dimension laws (each expected value recomputed independently):
//!   dim M_2(Gamma0(N)) = 2*g(X0(N)) + #cusps - 1,
//!   dim of the cuspidal subspace = 2*g(X0(N)).

use super::p1list::P1List;
use crate::cusps::Cusp;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::collections::{HashMap, HashSet};

/// Extended gcd on Integer: returns (g, x, y) with x*a + y*b = g = gcd(a,b) >= 0.
fn xgcd_integer(a: &Integer, b: &Integer) -> (Integer, Integer, Integer) {
    let (mut old_r, mut r) = (a.clone(), b.clone());
    let (mut old_s, mut s) = (Integer::one(), Integer::zero());
    let (mut old_t, mut t) = (Integer::zero(), Integer::one());
    while !r.is_zero() {
        let q = &old_r / &r; // truncated division keeps the Bezout invariant
        let new_r = &old_r - &(&q * &r);
        let new_s = &old_s - &(&q * &s);
        let new_t = &old_t - &(&q * &t);
        old_r = std::mem::replace(&mut r, new_r);
        old_s = std::mem::replace(&mut s, new_s);
        old_t = std::mem::replace(&mut t, new_t);
    }
    if old_r.signum() < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

/// A cusp as a coprime pair (a, c) meaning a/c, with infinity = (1, 0).
fn cusp_pair(x: &Cusp) -> (Integer, Integer) {
    match x {
        Cusp::Infinity => (Integer::one(), Integer::zero()),
        Cusp::Rational(p, q) => (p.clone(), q.clone()),
    }
}

/// Gamma0(N)-equivalence of cusps.
///
/// Criterion (Cremona, *Algorithms for Modular Elliptic Curves*, Prop.
/// 2.2.3; rederived here from the stabilizer of infinity): writing the
/// cusps as a1/c1, a2/c2 in lowest terms and choosing d_i with
/// a_i*d_i = 1 (mod c_i) (the bottom-right entry of an SL2(Z) matrix
/// g_i with g_i(infinity) = a_i/c_i), the cusps are equivalent under
/// Gamma0(N) if and only if
///     c2*d1 - c1*d2 = 0  (mod gcd(c1*c2, N)).
///
/// Derivation: every gamma with gamma(a1/c1) = a2/c2 has the form
/// +-g2*[[1,n],[0,1]]*g1^{-1}; its bottom-left entry is
/// c2*d1 - c1*d2 - n*c1*c2, and gamma is in Gamma0(N) for some n exactly
/// under the stated divisibility.  Verified constructively (the matrix
/// gamma is built and checked) for all cusp pairs at every level N <= 50.
pub fn cusps_equivalent_gamma0(n: u64, x: &Cusp, y: &Cusp) -> bool {
    assert!(n > 0, "level must be positive");
    let (a1, c1) = cusp_pair(x);
    let (a2, c2) = cusp_pair(y);
    let (g1, d1, _) = xgcd_integer(&a1, &c1);
    let (g2, d2, _) = xgcd_integer(&a2, &c2);
    assert!(
        g1.is_one() && g2.is_one(),
        "cusps must be in lowest terms (Cusp::new guarantees this)"
    );
    let nn = Integer::from(n as i64);
    let modulus = (&c1 * &c2).gcd(&nn);
    // c1*c2 >= 0 and n >= 1 imply modulus >= 1
    let lhs = &(&c2 * &d1) - &(&c1 * &d2);
    (&lhs % &modulus).is_zero()
}

/// Floor division a/b for Integer with b > 0.
fn div_floor(a: &Integer, b: &Integer) -> Integer {
    debug_assert!(b.signum() > 0);
    let q = a / b; // truncated
    let r = a - &(&q * b);
    if r.signum() < 0 {
        q - Integer::one()
    } else {
        q
    }
}

/// x mod n as an i64 in [0, n), for i64 n >= 1.
fn mod_n_i64(x: &Integer, n: i64) -> i64 {
    let nn = Integer::from(n);
    (x % &nn).to_i64().rem_euclid(n)
}

/// The space M_2(Gamma0(N)) of weight-2 modular symbols with trivial
/// character over Q, in the Manin-symbol presentation.
#[derive(Debug, Clone)]
pub struct ModularSymbolsGamma0 {
    level: u64,
    p1: P1List,
    dimension: usize,
    /// Indices (into the P^1 list) of the generators forming the basis of
    /// the quotient (the free columns of the reduced relation matrix).
    basis_indices: Vec<usize>,
    /// projection[i] = coordinates of the i-th Manin generator in the basis.
    projection: Vec<Vec<Rational>>,
    /// Representatives of the Gamma0(N)-classes of cusps.
    cusp_reps: Vec<Cusp>,
    /// boundary_gen[i] = image of the i-th generator under the boundary map
    /// delta(g{0,oo}) = [g(oo)] - [g(0)], as a vector over the cusp classes.
    boundary_gen: Vec<Vec<Rational>>,
    /// Boundary of each basis element (row per basis element).
    boundary_basis: Vec<Vec<Rational>>,
    /// Basis of the cuspidal subspace ker(delta), in basis coordinates.
    cuspidal_basis: Vec<Vec<Rational>>,
}

impl ModularSymbolsGamma0 {
    /// Build the space for level N (weight 2, trivial character, sign 0).
    pub fn new(n: u64) -> Self {
        let p1 = P1List::new(n);
        let ngen = p1.len();

        // --- relation matrix: x + xS = 0 and x + xT + xT^2 = 0 ---
        // Deduplicate orbits: the S-relation for i and for iS coincide, and
        // the T-relation is shared by the 3-cycle {i, iT, iT^2}.
        let mut rows: Vec<Vec<Rational>> = Vec::new();
        for i in 0..ngen {
            let si = p1.apply_s(i);
            if i <= si {
                let mut row = vec![Rational::zero(); ngen];
                row[i] = &row[i] + &Rational::one();
                row[si] = &row[si] + &Rational::one();
                rows.push(row);
            }
            let ti = p1.apply_t(i);
            let tti = p1.apply_t(ti);
            if i <= ti && i <= tti {
                let mut row = vec![Rational::zero(); ngen];
                row[i] = &row[i] + &Rational::one();
                row[ti] = &row[ti] + &Rational::one();
                row[tti] = &row[tti] + &Rational::one();
                rows.push(row);
            }
        }
        let nrows = rows.len();
        let flat: Vec<Rational> = rows.into_iter().flatten().collect();
        let mat = Matrix::from_vec(nrows, ngen, flat).expect("relation matrix shape");
        let rref = mat
            .reduced_row_echelon_form()
            .expect("exact rref over Q cannot fail");

        // --- quotient basis and projection ---
        let pivot_set: HashSet<usize> = rref.pivots.iter().copied().collect();
        let basis_indices: Vec<usize> =
            (0..ngen).filter(|j| !pivot_set.contains(j)).collect();
        let dimension = basis_indices.len();
        let pos_in_basis: HashMap<usize, usize> = basis_indices
            .iter()
            .enumerate()
            .map(|(k, &j)| (j, k))
            .collect();
        let mut projection = vec![vec![Rational::zero(); dimension]; ngen];
        for (j, row) in projection.iter_mut().enumerate() {
            if let Some(&k) = pos_in_basis.get(&j) {
                row[k] = Rational::one();
            }
        }
        for (r, &pc) in rref.pivots.iter().enumerate() {
            // rref row r reads: e_pc + sum_f rref[r][f] * e_f = 0 (f free),
            // so e_pc projects to -sum_f rref[r][f] * e_f.
            for (k, &f) in basis_indices.iter().enumerate() {
                let v = rref.matrix.get(r, f).expect("rref entry in range");
                if !v.is_zero() {
                    projection[pc][k] = -v;
                }
            }
        }

        // --- boundary map to the cusps ---
        let mut cusp_reps: Vec<Cusp> = Vec::new();
        let class_of = |cusp: Cusp, reps: &mut Vec<Cusp>| -> usize {
            for (idx, r) in reps.iter().enumerate() {
                if cusps_equivalent_gamma0(n, &cusp, r) {
                    return idx;
                }
            }
            reps.push(cusp);
            reps.len() - 1
        };
        let mut gen_classes: Vec<(usize, usize)> = Vec::with_capacity(ngen);
        for i in 0..ngen {
            let m = p1.lift_to_sl2z(i);
            // g{0, oo}: boundary = [g(oo)] - [g(0)] = [a/c] - [b/d]
            let plus = Cusp::from_i64(m[0][0], m[1][0]);
            let minus = Cusp::from_i64(m[0][1], m[1][1]);
            let cp = class_of(plus, &mut cusp_reps);
            let cm = class_of(minus, &mut cusp_reps);
            gen_classes.push((cp, cm));
        }
        let nclasses = cusp_reps.len();
        let mut boundary_gen = vec![vec![Rational::zero(); nclasses]; ngen];
        for (i, &(cp, cm)) in gen_classes.iter().enumerate() {
            boundary_gen[i][cp] = &boundary_gen[i][cp] + &Rational::one();
            boundary_gen[i][cm] = &boundary_gen[i][cm] - &Rational::one();
        }
        let boundary_basis: Vec<Vec<Rational>> = basis_indices
            .iter()
            .map(|&j| boundary_gen[j].clone())
            .collect();

        // --- cuspidal subspace = kernel of the boundary map ---
        let cuspidal_basis = if dimension == 0 {
            Vec::new()
        } else {
            // A is nclasses x dimension with A[c][k] = delta(basis_k)[c];
            // kernel(A) = { x : sum_k x_k * delta(basis_k) = 0 }.
            let mut flat = Vec::with_capacity(nclasses * dimension);
            for c in 0..nclasses {
                for row in boundary_basis.iter() {
                    flat.push(row[c].clone());
                }
            }
            let a = Matrix::from_vec(nclasses, dimension, flat)
                .expect("boundary matrix shape");
            a.kernel().expect("exact kernel over Q cannot fail")
        };

        ModularSymbolsGamma0 {
            level: n,
            p1,
            dimension,
            basis_indices,
            projection,
            cusp_reps,
            boundary_gen,
            boundary_basis,
            cuspidal_basis,
        }
    }

    /// The level N.
    pub fn level(&self) -> u64 {
        self.level
    }

    /// The underlying projective line P^1(Z/NZ) indexing the generators.
    pub fn p1(&self) -> &P1List {
        &self.p1
    }

    /// Number of Manin generators, |P^1(Z/NZ)|.
    pub fn num_generators(&self) -> usize {
        self.p1.len()
    }

    /// dim M_2(Gamma0(N)) = 2*g(X0(N)) + #cusps - 1.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Indices into the P^1 list of the generators whose images form the
    /// basis of the quotient.
    pub fn basis_manin_indices(&self) -> &[usize] {
        &self.basis_indices
    }

    /// The (c : d) pair of the i-th Manin generator.
    pub fn manin_generator(&self, i: usize) -> (u64, u64) {
        self.p1.list()[i]
    }

    /// Coordinates of the i-th Manin generator in the quotient basis.
    pub fn manin_generator_coords(&self, i: usize) -> &[Rational] {
        &self.projection[i]
    }

    /// Number of Gamma0(N)-classes of cusps.
    pub fn cusp_class_count(&self) -> usize {
        self.cusp_reps.len()
    }

    /// Representatives of the cusp classes.
    pub fn cusp_representatives(&self) -> &[Cusp] {
        &self.cusp_reps
    }

    /// Index of the cusp class containing the given cusp.
    pub fn cusp_class_index(&self, cusp: &Cusp) -> usize {
        for (idx, r) in self.cusp_reps.iter().enumerate() {
            if cusps_equivalent_gamma0(self.level, cusp, r) {
                return idx;
            }
        }
        unreachable!("every cusp is equivalent to a representative")
    }

    /// Boundary of the i-th Manin generator, delta(g{0,oo}) = [g(oo)] - [g(0)],
    /// as a vector over the cusp classes.
    pub fn boundary_of_generator(&self, i: usize) -> &[Rational] {
        &self.boundary_gen[i]
    }

    /// Boundary of each basis element (one row per basis element, columns
    /// indexed by cusp classes).
    pub fn boundary_of_basis(&self) -> &[Vec<Rational>] {
        &self.boundary_basis
    }

    /// Boundary of an element given in basis coordinates.
    pub fn boundary_of(&self, coords: &[Rational]) -> Vec<Rational> {
        assert_eq!(coords.len(), self.dimension, "coordinate length");
        let mut out = vec![Rational::zero(); self.cusp_reps.len()];
        for (x, row) in coords.iter().zip(self.boundary_basis.iter()) {
            if x.is_zero() {
                continue;
            }
            for (o, v) in out.iter_mut().zip(row.iter()) {
                *o = &*o + &(x * v);
            }
        }
        out
    }

    /// True iff the element (in basis coordinates) lies in the cuspidal
    /// subspace, i.e. has zero boundary.
    pub fn is_cuspidal(&self, coords: &[Rational]) -> bool {
        self.boundary_of(coords).iter().all(|v| v.is_zero())
    }

    /// dim of the cuspidal subspace = 2*g(X0(N)).
    pub fn cuspidal_dimension(&self) -> usize {
        self.cuspidal_basis.len()
    }

    /// Basis of the cuspidal subspace, in ambient basis coordinates.
    pub fn cuspidal_basis(&self) -> &[Vec<Rational>] {
        &self.cuspidal_basis
    }

    /// The modular symbol {oo, a/b} as basis coordinates, via the Manin
    /// trick: if p_k/q_k are the continued-fraction convergents of a/b
    /// (with p_{-1}/q_{-1} = oo), then
    ///   {oo, a/b} = sum_k g_k{0, oo},
    ///   g_k = [[(-1)^{k-1} p_k, p_{k-1}], [(-1)^{k-1} q_k, q_{k-1}]]
    /// which is the Manin generator ((-1)^{k-1} q_k : q_{k-1}).
    fn symbol_from_infinity(&self, cusp: &Cusp) -> Vec<Rational> {
        let mut acc = vec![Rational::zero(); self.dimension];
        let (mut a, mut b) = match cusp {
            Cusp::Infinity => return acc,
            Cusp::Rational(p, q) => (p.clone(), q.clone()),
        };
        if b.signum() < 0 {
            a = -a;
            b = -b;
        }
        let n = self.level as i64;
        let (mut q_km2, mut q_km1) = (Integer::one(), Integer::zero());
        let mut sign = -1i64; // (-1)^{k-1} for k = 0
        while !b.is_zero() {
            let ak = div_floor(&a, &b);
            let r = &a - &(&ak * &b);
            let qk = &(&ak * &q_km1) + &q_km2;
            let c = sign * mod_n_i64(&qk, n);
            let d = mod_n_i64(&q_km1, n);
            let idx = self
                .p1
                .index_of(c, d)
                .expect("consecutive convergent denominators are coprime");
            for (t, v) in acc.iter_mut().zip(self.projection[idx].iter()) {
                *t = &*t + v;
            }
            q_km2 = std::mem::replace(&mut q_km1, qk);
            a = std::mem::replace(&mut b, r);
            sign = -sign;
        }
        acc
    }

    /// The modular symbol {alpha, beta} expressed in basis coordinates,
    /// via {alpha, beta} = {oo, beta} - {oo, alpha} and the Manin trick.
    pub fn modular_symbol(&self, alpha: &Cusp, beta: &Cusp) -> Vec<Rational> {
        let to = self.symbol_from_infinity(beta);
        let from = self.symbol_from_infinity(alpha);
        to.iter()
            .zip(from.iter())
            .map(|(t, f)| t - f)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abvar::J0;
    use crate::arithgroup::Gamma0;

    /// Genus of X0(N) for N = 1..=50, recomputed independently in python via
    /// g = 1 + mu/12 - nu2/4 - nu3/3 - cusps/2 (mu = index, nu2/nu3 counts of
    /// solutions of x^2+1=0 / x^2+x+1=0 mod N).  Spot values agree with the
    /// literature: g(11)=1, g(22)=2, g(23)=2, g(37)=2, g(50)=2.
    const GENUS_X0: [usize; 50] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 2,
        1, 0, 2, 1, 2, 2, 3, 2, 1, 3, 3, 3, 1, 2, 4, 3, 3, 3, 5, 3, 4, 3, 5,
        4, 3, 1, 2,
    ];

    /// #cusps of Gamma0(N) for N = 1..=50, recomputed independently in
    /// python via sum_{d|N} phi(gcd(d, N/d)).
    const CUSPS_X0: [usize; 50] = [
        1, 2, 2, 3, 2, 4, 2, 4, 4, 4, 2, 6, 2, 4, 4, 6, 2, 8, 2, 6, 4, 4, 2,
        8, 6, 4, 6, 6, 2, 8, 2, 8, 4, 4, 4, 12, 2, 4, 4, 8, 2, 8, 2, 6, 8, 4,
        2, 12, 8, 12,
    ];

    fn check_stage1_gates(n: u64) {
        let m = ModularSymbolsGamma0::new(n);
        let i = (n - 1) as usize;
        // the crate's own genus and cusp count must agree with the
        // independent python recomputation (no single-source confirmation)
        assert_eq!(J0::new(n).dimension(), GENUS_X0[i], "crate genus_x0({n}) vs python");
        assert_eq!(
            Gamma0::new(n).compute_cusp_count() as usize,
            CUSPS_X0[i],
            "crate cusp count({n}) vs python"
        );
        let (g, c) = (GENUS_X0[i], CUSPS_X0[i]);
        // DIMENSION LAW: dim M_2(Gamma0(N)) = 2g + #cusps - 1
        assert_eq!(m.dimension(), 2 * g + c - 1, "dim M_2(Gamma0({n}))");
        // cusp classes found by the equivalence criterion match the formula
        assert_eq!(m.cusp_class_count(), c, "cusp classes at level {n}");
        // CUSPIDAL DIMENSION GATE: dim S-part = 2g
        assert_eq!(m.cuspidal_dimension(), 2 * g, "cuspidal dim at level {n}");

        // structural consistency at every level:
        let p1 = m.p1();
        let dim = m.dimension();
        let zero_vec = vec![Rational::zero(); dim];
        let zero_bnd = vec![Rational::zero(); m.cusp_class_count()];
        let add = |a: &[Rational], b: &[Rational]| -> Vec<Rational> {
            a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
        };
        for i in 0..m.num_generators() {
            let si = p1.apply_s(i);
            let ti = p1.apply_t(i);
            let tti = p1.apply_t(ti);
            // projection respects the relations: x + xS = 0, x + xT + xT^2 = 0
            assert_eq!(
                add(m.manin_generator_coords(i), m.manin_generator_coords(si)),
                zero_vec,
                "S-relation in quotient at level {n}, generator {i}"
            );
            assert_eq!(
                add(
                    &add(m.manin_generator_coords(i), m.manin_generator_coords(ti)),
                    m.manin_generator_coords(tti)
                ),
                zero_vec,
                "T-relation in quotient at level {n}, generator {i}"
            );
            // the boundary map kills the relations (so it is well defined
            // on the quotient)
            assert_eq!(
                add(m.boundary_of_generator(i), m.boundary_of_generator(si)),
                zero_bnd,
                "boundary kills S-relation at level {n}"
            );
            assert_eq!(
                add(
                    &add(m.boundary_of_generator(i), m.boundary_of_generator(ti)),
                    m.boundary_of_generator(tti)
                ),
                zero_bnd,
                "boundary kills T-relation at level {n}"
            );
        }
        // every cuspidal basis vector has zero boundary
        for v in m.cuspidal_basis() {
            assert!(m.is_cuspidal(v), "cuspidal basis vector at level {n}");
        }
    }

    #[test]
    fn test_stage1_gates_levels_1_to_30() {
        for n in 1..=30 {
            check_stage1_gates(n);
        }
    }

    #[test]
    fn test_stage1_gates_levels_31_to_50() {
        for n in 31..=50 {
            check_stage1_gates(n);
        }
    }

    #[test]
    fn test_explicit_landmark_levels() {
        // N = 11: first genus-1 level. dim M = 3, cuspidal 2, 2 cusps.
        let m11 = ModularSymbolsGamma0::new(11);
        assert_eq!(m11.dimension(), 3);
        assert_eq!(m11.cuspidal_dimension(), 2);
        assert_eq!(m11.cusp_class_count(), 2);
        // N = 22: genus 2 (verified independently: mu=36, nu2=nu3=0,
        // cusps=4 => g = 1 + 3 - 2 = 2), so cuspidal dim 4.
        let m22 = ModularSymbolsGamma0::new(22);
        assert_eq!(m22.cuspidal_dimension(), 4);
        // N = 37: genus 2, cuspidal dim 4.
        let m37 = ModularSymbolsGamma0::new(37);
        assert_eq!(m37.cuspidal_dimension(), 4);
        assert_eq!(m37.dimension(), 5);
        // N = 1: X0(1) = P^1, one cusp: the space is zero.
        let m1 = ModularSymbolsGamma0::new(1);
        assert_eq!(m1.dimension(), 0);
        assert_eq!(m1.cuspidal_dimension(), 0);
        assert_eq!(m1.cusp_class_count(), 1);
    }

    #[test]
    fn test_cusp_equivalence_basic() {
        // N = 11: cusps 0 and oo are inequivalent; 1/11 ~ oo, 2/1 ~ 0
        assert!(!cusps_equivalent_gamma0(11, &Cusp::zero(), &Cusp::infinity()));
        assert!(cusps_equivalent_gamma0(
            11,
            &Cusp::from_i64(1, 11),
            &Cusp::infinity()
        ));
        assert!(cusps_equivalent_gamma0(
            11,
            &Cusp::from_i64(2, 1),
            &Cusp::zero()
        ));
        // N = 4: classes are {0, 1/2, oo}; 3/2 ~ 1/2, 1/4 ~ oo
        assert!(cusps_equivalent_gamma0(
            4,
            &Cusp::from_i64(3, 2),
            &Cusp::from_i64(1, 2)
        ));
        assert!(cusps_equivalent_gamma0(
            4,
            &Cusp::from_i64(1, 4),
            &Cusp::infinity()
        ));
        assert!(!cusps_equivalent_gamma0(
            4,
            &Cusp::from_i64(1, 2),
            &Cusp::zero()
        ));
        assert!(!cusps_equivalent_gamma0(
            4,
            &Cusp::from_i64(1, 2),
            &Cusp::infinity()
        ));
        // N = 1: everything is equivalent
        assert!(cusps_equivalent_gamma0(1, &Cusp::zero(), &Cusp::infinity()));
        // equivalence is reflexive and symmetric on a sample
        for n in [2u64, 6, 11, 12, 25] {
            for c1 in [Cusp::zero(), Cusp::infinity(), Cusp::from_i64(1, 2)] {
                assert!(cusps_equivalent_gamma0(n, &c1, &c1));
                for c2 in [Cusp::zero(), Cusp::infinity(), Cusp::from_i64(2, 3)] {
                    assert_eq!(
                        cusps_equivalent_gamma0(n, &c1, &c2),
                        cusps_equivalent_gamma0(n, &c2, &c1),
                        "symmetry at level {n}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_manin_trick_basic_identities() {
        for n in [2u64, 11, 14, 15, 24, 37] {
            let m = ModularSymbolsGamma0::new(n);
            // {0, oo} is the identity-coset Manin symbol (0 : 1)
            let zero_inf = m.modular_symbol(&Cusp::zero(), &Cusp::infinity());
            let idx01 = m.p1().index_of(0, 1).unwrap();
            assert_eq!(zero_inf.as_slice(), m.manin_generator_coords(idx01));
            // {oo, 0} is the S-coset Manin symbol (1 : 0), and = -{0, oo}
            let inf_zero = m.modular_symbol(&Cusp::infinity(), &Cusp::zero());
            let idx10 = m.p1().index_of(1, 0).unwrap();
            assert_eq!(inf_zero.as_slice(), m.manin_generator_coords(idx10));
            let neg: Vec<Rational> = zero_inf.iter().map(|v| -v).collect();
            assert_eq!(inf_zero, neg);
            // {0, 1/N} = {0, oo} since [[1,0],[N,1]] in Gamma0(N) maps one
            // path to the other
            let s = m.modular_symbol(&Cusp::zero(), &Cusp::from_i64(1, n as i64));
            assert_eq!(s, zero_inf, "{{0, 1/{n}}} = {{0, oo}} at level {n}");
            // {a, a} = 0
            for a in [
                Cusp::zero(),
                Cusp::infinity(),
                Cusp::from_i64(1, 2),
                Cusp::from_i64(3, 7),
                Cusp::from_i64(-2, 5),
            ] {
                let v = m.modular_symbol(&a, &a);
                assert!(v.iter().all(|x| x.is_zero()), "{{a, a}} = 0 at level {n}");
            }
            // {a,b} + {b,c} = {a,c}
            let (a, b, c) = (
                Cusp::from_i64(-1, 3),
                Cusp::from_i64(2, 7),
                Cusp::infinity(),
            );
            let ab = m.modular_symbol(&a, &b);
            let bc = m.modular_symbol(&b, &c);
            let ac = m.modular_symbol(&a, &c);
            let sum: Vec<Rational> =
                ab.iter().zip(bc.iter()).map(|(x, y)| x + y).collect();
            assert_eq!(sum, ac, "additivity at level {n}");
        }
    }

    /// gamma(cusp) for gamma = [[a,b],[c,d]] acting as a Mobius map.
    fn apply_gamma(g: [[i64; 2]; 2], cusp: &Cusp) -> Cusp {
        let (p, q) = match cusp {
            Cusp::Infinity => (Integer::one(), Integer::zero()),
            Cusp::Rational(p, q) => (p.clone(), q.clone()),
        };
        let a = Integer::from(g[0][0]);
        let b = Integer::from(g[0][1]);
        let c = Integer::from(g[1][0]);
        let d = Integer::from(g[1][1]);
        Cusp::new(&(&a * &p) + &(&b * &q), &(&c * &p) + &(&d * &q))
    }

    fn mat_mul(x: [[i64; 2]; 2], y: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
        [
            [
                x[0][0] * y[0][0] + x[0][1] * y[1][0],
                x[0][0] * y[0][1] + x[0][1] * y[1][1],
            ],
            [
                x[1][0] * y[0][0] + x[1][1] * y[1][0],
                x[1][0] * y[0][1] + x[1][1] * y[1][1],
            ],
        ]
    }

    #[test]
    fn test_gamma0_invariance_of_modular_symbols() {
        // The deep consistency test: {gamma(a), gamma(b)} = {a, b} in the
        // quotient for gamma in Gamma0(N).  This exercises the P^1
        // normalization, the relations, the projection and the Manin trick
        // all at once.
        for n in [11u64, 15, 24, 33] {
            let m = ModularSymbolsGamma0::new(n);
            let ni = n as i64;
            let t: [[i64; 2]; 2] = [[1, 1], [0, 1]];
            let tinv: [[i64; 2]; 2] = [[1, -1], [0, 1]];
            let v: [[i64; 2]; 2] = [[1, 0], [ni, 1]];
            let vinv: [[i64; 2]; 2] = [[1, 0], [-ni, 1]];
            let gens = [t, tinv, v, vinv];
            let pairs = [
                (Cusp::zero(), Cusp::infinity()),
                (Cusp::from_i64(1, 2), Cusp::from_i64(2, 3)),
                (Cusp::from_i64(-1, 3), Cusp::infinity()),
            ];
            // all words of length <= 3 in the generators
            let mut words: Vec<[[i64; 2]; 2]> = vec![[[1, 0], [0, 1]]];
            let mut layer = vec![[[1i64, 0], [0, 1]]];
            for _ in 0..3 {
                let mut next = Vec::new();
                for w in &layer {
                    for g in &gens {
                        next.push(mat_mul(*w, *g));
                    }
                }
                words.extend(next.iter().copied());
                layer = next;
            }
            for w in &words {
                assert_eq!(w[0][0] * w[1][1] - w[0][1] * w[1][0], 1, "det 1");
                assert_eq!(w[1][0].rem_euclid(ni), 0, "gamma in Gamma0({n})");
                for (a, b) in &pairs {
                    let lhs =
                        m.modular_symbol(&apply_gamma(*w, a), &apply_gamma(*w, b));
                    let rhs = m.modular_symbol(a, b);
                    assert_eq!(lhs, rhs, "Gamma0({n})-invariance failed");
                }
            }
        }
    }

    #[test]
    fn test_boundary_of_winding_path() {
        // At level 11 the path {0, oo} joins the two cusp classes, so its
        // boundary is [oo] - [0] != 0 and it is not cuspidal.
        let m = ModularSymbolsGamma0::new(11);
        let w = m.modular_symbol(&Cusp::zero(), &Cusp::infinity());
        assert!(!m.is_cuspidal(&w));
        let b = m.boundary_of(&w);
        let cls_inf = m.cusp_class_index(&Cusp::infinity());
        let cls_zero = m.cusp_class_index(&Cusp::zero());
        assert_ne!(cls_inf, cls_zero);
        assert_eq!(b[cls_inf], Rational::one());
        assert_eq!(b[cls_zero], -&Rational::one());
        for (i, v) in b.iter().enumerate() {
            if i != cls_inf && i != cls_zero {
                assert!(v.is_zero());
            }
        }
    }
}
