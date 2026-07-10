//! Classical bases of symmetric functions and the change-of-basis engine.
//!
//! Ports the mathematics of MAGMA Handbook **Chapter 146 — Symmetric Functions**
//! (Macdonald, *Symmetric functions and Hall polynomials* [Mac95]). It supplies the
//! five classical bases indexed by partitions — monomial `m`, elementary `e`,
//! (complete) homogeneous `h`, power sum `p`, and Schur `s` — together with the
//! rational transition matrices between them (§146.5), the Hall inner product
//! (§146.4.10), the omega/Frobenius involution (§146.4.9), plethysm (§146.4.5) and the
//! Jacobi–Trudi determinants for the Schur functions.
//!
//! Design: the power-sum basis is a free polynomial algebra over `Q` on
//! `p_1, p_2, ...`, so it is used as the internal computational hub. Every basis has a
//! known rational expansion into `p`; all conversions compose through `p` via
//! `Matrix<Rational>` linear algebra (rustmath-matrix). The matrices among the
//! `s, h, m, e` bases are integer-valued; any change to/from the `p` basis is over the
//! rationals ([Mac95], pp. 54–58).

use crate::{SymFun, SymmetricFunctionBasis};
use rustmath_combinatorics::{partitions, Partition};
use rustmath_core::Ring;
use rustmath_matrix::Matrix;
use rustmath_rationals::Rational;
use std::collections::{HashMap, HashSet};

/// A linear combination of power-sum functions `p_lambda`, keyed by partition.
pub type PMap = HashMap<Partition, Rational>;

/// The five classical bases of the algebra of symmetric functions (MAGMA ch 146).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClassicalBasis {
    /// Monomial `m_lambda`.
    Monomial,
    /// Elementary `e_lambda`.
    Elementary,
    /// Complete homogeneous `h_lambda`.
    Homogeneous,
    /// Power sum `p_lambda`.
    PowerSum,
    /// Schur `s_lambda`.
    Schur,
}

impl ClassicalBasis {
    /// Map from the (four-variant) `SymFun` basis tag to a classical basis.
    pub fn from_symfun(b: SymmetricFunctionBasis) -> ClassicalBasis {
        match b {
            SymmetricFunctionBasis::Monomial => ClassicalBasis::Monomial,
            SymmetricFunctionBasis::Elementary => ClassicalBasis::Elementary,
            SymmetricFunctionBasis::PowerSum => ClassicalBasis::PowerSum,
            SymmetricFunctionBasis::Schur => ClassicalBasis::Schur,
        }
    }

    /// Map a classical basis back to a `SymFun` basis tag, if representable
    /// (the `SymFun` type has no dedicated homogeneous variant).
    pub fn to_symfun(self) -> Option<SymmetricFunctionBasis> {
        match self {
            ClassicalBasis::Monomial => Some(SymmetricFunctionBasis::Monomial),
            ClassicalBasis::Elementary => Some(SymmetricFunctionBasis::Elementary),
            ClassicalBasis::PowerSum => Some(SymmetricFunctionBasis::PowerSum),
            ClassicalBasis::Schur => Some(SymmetricFunctionBasis::Schur),
            ClassicalBasis::Homogeneous => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

#[inline]
fn rati(n: i64) -> Rational {
    Rational::from(n)
}

/// `1/z` as an exact rational.
#[inline]
fn rrecip(z: i64) -> Rational {
    Rational::new(1i64, z).expect("nonzero denominator")
}

/// The order of the centraliser `z_lambda = prod_i (m_i! * i^{m_i})` where `m_i`
/// is the multiplicity of the part `i`. Equals `n!/|C_lambda|`'s reciprocal role:
/// `|conjugacy class| = n!/z_lambda`. Also `<p_lambda, p_lambda> = z_lambda`.
pub fn centralizer_size(lambda: &Partition) -> i64 {
    let mut mult: HashMap<usize, u32> = HashMap::new();
    for &part in lambda.parts() {
        *mult.entry(part).or_insert(0) += 1;
    }
    let mut z: i64 = 1;
    for (part, m) in mult {
        for i in 1..=m as i64 {
            z *= i;
        }
        z *= (part as i64).pow(m);
    }
    z
}

// ---------------------------------------------------------------------------
// Power-sum arithmetic (the internal hub ring)
// ---------------------------------------------------------------------------

fn padd(map: &mut PMap, part: Partition, c: Rational) {
    if c.is_zero() {
        return;
    }
    let entry = map.entry(part.clone()).or_insert_with(Rational::zero);
    *entry = entry.clone() + c;
    if entry.is_zero() {
        map.remove(&part);
    }
}

/// Multiply two power-sum combinations: `p_lambda * p_mu = p_{lambda cup mu}`.
fn pmul(a: &PMap, b: &PMap) -> PMap {
    let mut out = PMap::new();
    for (la, ca) in a {
        for (mu, cb) in b {
            let mut parts = la.parts().to_vec();
            parts.extend_from_slice(mu.parts());
            padd(&mut out, Partition::new(parts), ca.clone() * cb.clone());
        }
    }
    out
}

fn p_one() -> PMap {
    let mut m = PMap::new();
    m.insert(Partition::new(vec![]), Rational::one());
    m
}

// ---------------------------------------------------------------------------
// Single-basis-element expansions into the power-sum hub
// ---------------------------------------------------------------------------

/// `h_n = sum_{rho |- n} p_rho / z_rho` (generating-function identity, [Mac95]).
fn h_n_to_p(n: usize) -> PMap {
    if n == 0 {
        return p_one();
    }
    let mut m = PMap::new();
    for rho in partitions(n) {
        let z = centralizer_size(&rho);
        m.insert(rho, rrecip(z));
    }
    m
}

/// `e_n = sum_{rho |- n} eps_rho p_rho / z_rho`, `eps_rho = (-1)^{n - l(rho)}`.
fn e_n_to_p(n: usize) -> PMap {
    if n == 0 {
        return p_one();
    }
    let mut m = PMap::new();
    for rho in partitions(n) {
        let z = centralizer_size(&rho);
        let sign = if (n - rho.length()) % 2 == 0 { 1 } else { -1 };
        m.insert(rho.clone(), rrecip(z) * rati(sign));
    }
    m
}

/// `h_lambda = prod_i h_{lambda_i}` expanded in the power-sum basis.
fn h_partition_to_p(lambda: &Partition) -> PMap {
    let mut acc = p_one();
    for &part in lambda.parts() {
        acc = pmul(&acc, &h_n_to_p(part));
    }
    acc
}

/// `e_lambda = prod_i e_{lambda_i}` expanded in the power-sum basis.
fn e_partition_to_p(lambda: &Partition) -> PMap {
    let mut acc = p_one();
    for &part in lambda.parts() {
        acc = pmul(&acc, &e_n_to_p(part));
    }
    acc
}

/// `s_lambda = sum_{rho |- n} chi^lambda(rho) p_rho / z_rho` (via the character table).
fn s_partition_to_p(lambda: &Partition) -> PMap {
    let n = lambda.sum();
    let mut m = PMap::new();
    if n == 0 {
        return p_one();
    }
    for rho in partitions(n) {
        let chi = mn_character(lambda, &rho);
        if chi != 0 {
            let z = centralizer_size(&rho);
            m.insert(rho.clone(), rrecip(z) * rati(chi));
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Murnaghan–Nakayama rule (correct abacus / beta-set implementation)
// ---------------------------------------------------------------------------

/// Border-strip removals of size `k` from the partition `lambda` (given as its
/// parts). Returns `(inner_partition_parts, leg_length)` pairs, where the
/// Murnaghan–Nakayama sign of the removed strip is `(-1)^leg_length`.
///
/// Uses the James–Kerber 1-runner abacus: with beta-set
/// `beta_i = lambda_i + (m-1-i)` (`m` = number of parts), a size-`k` border strip
/// corresponds to sliding a bead from position `b` to the empty position `b-k`;
/// the leg length is the number of beads strictly between `b-k` and `b`.
pub fn border_strip_removals(lambda: &[usize], k: usize) -> Vec<(Vec<usize>, usize)> {
    let m = lambda.len();
    if m == 0 || k == 0 {
        return Vec::new();
    }
    let beta: Vec<i64> = (0..m)
        .map(|i| lambda[i] as i64 + (m - 1 - i) as i64)
        .collect();
    let bset: HashSet<i64> = beta.iter().copied().collect();
    let mut out = Vec::new();
    for i in 0..m {
        let b = beta[i];
        let c = b - k as i64;
        if c < 0 || bset.contains(&c) {
            continue;
        }
        let leg = beta.iter().filter(|&&x| x > c && x < b).count();
        let mut nb = beta.clone();
        nb[i] = c;
        nb.sort_unstable_by(|a, b| b.cmp(a));
        let mut inner: Vec<usize> = (0..m).map(|j| (nb[j] - (m - 1 - j) as i64) as usize).collect();
        while let Some(&0) = inner.last() {
            inner.pop();
        }
        out.push((inner, leg));
    }
    out
}

/// The irreducible symmetric-group character value `chi^lambda(rho)` via the
/// Murnaghan–Nakayama rule. `lambda` indexes the irreducible representation,
/// `rho` is the cycle type. Correct replacement for the buggy combinatorics-backed
/// version (which over-counted duplicate inner partitions).
pub fn mn_character(lambda: &Partition, rho: &Partition) -> i64 {
    if lambda.sum() != rho.sum() {
        return 0;
    }
    mn_rec(lambda.parts(), rho.parts())
}

fn mn_rec(lambda: &[usize], rho: &[usize]) -> i64 {
    if lambda.is_empty() {
        return if rho.is_empty() { 1 } else { 0 };
    }
    if rho.is_empty() {
        return 0;
    }
    let k = rho[0];
    let rest = &rho[1..];
    let mut total = 0i64;
    for (inner, leg) in border_strip_removals(lambda, k) {
        let sign = if leg % 2 == 0 { 1 } else { -1 };
        total += sign * mn_rec(&inner, rest);
    }
    total
}

// ---------------------------------------------------------------------------
// Per-degree transition tables
// ---------------------------------------------------------------------------

/// The `X -> PowerSum` matrices for every classical basis at a fixed degree `n`.
/// Rows and columns are indexed by `partitions(n)` in a fixed internal order; entry
/// `(i,j)` of the `X` matrix is the coefficient of `p_{parts[j]}` in `x_{parts[i]}`.
struct DegTables {
    parts: Vec<Partition>,
    index: HashMap<Partition, usize>,
    h: Matrix<Rational>,
    e: Matrix<Rational>,
    s: Matrix<Rational>,
    m: Matrix<Rational>,
    p: Matrix<Rational>,
}

fn pmap_to_row(pm: &PMap, index: &HashMap<Partition, usize>, len: usize) -> Vec<Rational> {
    let mut row = vec![Rational::zero(); len];
    for (part, c) in pm {
        if let Some(&j) = index.get(part) {
            row[j] = c.clone();
        }
    }
    row
}

fn matrix_from_rows(rows: Vec<Vec<Rational>>, n: usize) -> Matrix<Rational> {
    let mut data = Vec::with_capacity(n * n);
    for r in rows {
        data.extend(r);
    }
    Matrix::from_vec(n, n, data).expect("square matrix")
}

impl DegTables {
    fn build(n: usize) -> DegTables {
        let parts = partitions(n);
        let len = parts.len();
        let mut index = HashMap::new();
        for (i, p) in parts.iter().enumerate() {
            index.insert(p.clone(), i);
        }

        let h_rows: Vec<Vec<Rational>> = parts
            .iter()
            .map(|lam| pmap_to_row(&h_partition_to_p(lam), &index, len))
            .collect();
        let e_rows: Vec<Vec<Rational>> = parts
            .iter()
            .map(|lam| pmap_to_row(&e_partition_to_p(lam), &index, len))
            .collect();
        let s_rows: Vec<Vec<Rational>> = parts
            .iter()
            .map(|lam| pmap_to_row(&s_partition_to_p(lam), &index, len))
            .collect();

        let h = matrix_from_rows(h_rows, len);
        let e = matrix_from_rows(e_rows, len);
        let s = matrix_from_rows(s_rows, len);
        let p = Matrix::identity(len);

        // m -> p : M_M = (M_H^{-1})^T * diag(1/z_rho).
        // Derivation: <m_lambda, h_mu> = delta and <p_rho, p_sigma> = z_rho delta
        // give M_M * diag(z) * M_H^T = I.
        let h_inv = h
            .inverse()
            .expect("invertible")
            .expect("h->p is nonsingular");
        let h_inv_t = h_inv.transpose();
        let mut m_data = Vec::with_capacity(len * len);
        for i in 0..len {
            for j in 0..len {
                // (M_H^{-1})^T_{i,j} * (1/z_{parts[j]})
                let v = h_inv_t.get(i, j).expect("in range").clone();
                let z = centralizer_size(&parts[j]);
                m_data.push(v * rrecip(z));
            }
        }
        let m = Matrix::from_vec(len, len, m_data).expect("square");

        DegTables {
            parts,
            index,
            h,
            e,
            s,
            m,
            p,
        }
    }

    fn matrix_of(&self, basis: ClassicalBasis) -> &Matrix<Rational> {
        match basis {
            ClassicalBasis::Homogeneous => &self.h,
            ClassicalBasis::Elementary => &self.e,
            ClassicalBasis::Schur => &self.s,
            ClassicalBasis::Monomial => &self.m,
            ClassicalBasis::PowerSum => &self.p,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition matrices (MAGMA §146.5)
// ---------------------------------------------------------------------------

/// The change-of-basis matrix expressing `from`-basis functions of weight `n` in
/// the `to`-basis: row `i` (partition `parts[i]`) holds the `to`-basis coefficients
/// of the `from`-basis element indexed by `parts[i]`. Rows/columns are indexed by
/// the internal `partitions(n)` order; see [`partitions_order`].
///
/// Realises the `XToYMatrix(n)` family of MAGMA ch 146.5 over `Rational`.
pub fn transition_matrix(
    from: ClassicalBasis,
    to: ClassicalBasis,
    n: usize,
) -> Matrix<Rational> {
    if n == 0 {
        // Only the empty partition; every basis element is the scalar 1.
        return Matrix::identity(1);
    }
    let t = DegTables::build(n);
    let m_from = t.matrix_of(from);
    let m_to = t.matrix_of(to);
    let m_to_inv = m_to
        .inverse()
        .expect("invertible")
        .expect("nonsingular basis matrix");
    // X -> Y = M_X * M_Y^{-1}.
    m_from.mul(&m_to_inv).expect("compatible dims")
}

/// The internal partition ordering used for rows/columns of transition matrices.
pub fn partitions_order(n: usize) -> Vec<Partition> {
    partitions(n)
}

// ---------------------------------------------------------------------------
// Change of basis for whole symmetric functions
// ---------------------------------------------------------------------------

fn coeff_map_in_basis(sym: &SymFun, target: ClassicalBasis) -> HashMap<Partition, Rational> {
    let from = ClassicalBasis::from_symfun(sym.basis);
    let mut result: HashMap<Partition, Rational> = HashMap::new();

    // Group support by degree, convert degree by degree.
    let mut by_degree: HashMap<usize, Vec<(Partition, Rational)>> = HashMap::new();
    for (part, c) in &sym.coeffs {
        if c.is_zero() {
            continue;
        }
        by_degree
            .entry(part.sum())
            .or_default()
            .push((part.clone(), c.clone()));
    }

    for (n, terms) in by_degree {
        if n == 0 {
            // Scalar term: identical in every basis.
            for (part, c) in terms {
                let e = result.entry(part).or_insert_with(Rational::zero);
                *e = e.clone() + c;
            }
            continue;
        }
        if from == target {
            for (part, c) in terms {
                let e = result.entry(part).or_insert_with(Rational::zero);
                *e = e.clone() + c;
            }
            continue;
        }
        let t = DegTables::build(n);
        let m_from = t.matrix_of(from);
        let m_to = t.matrix_of(target);
        let m_to_inv = m_to
            .inverse()
            .expect("invertible")
            .expect("nonsingular");
        let x_to_y = m_from.mul(&m_to_inv).expect("dims");
        let len = t.parts.len();
        // w_j = sum_i c_i * XToY[i][j]
        for (part, c) in terms {
            let i = *t.index.get(&part).expect("partition of n");
            for j in 0..len {
                let entry = x_to_y.get(i, j).expect("range").clone();
                if entry.is_zero() {
                    continue;
                }
                let contrib = c.clone() * entry;
                let e = result
                    .entry(t.parts[j].clone())
                    .or_insert_with(Rational::zero);
                *e = e.clone() + contrib;
            }
        }
    }

    result.retain(|_, c| !c.is_zero());
    result
}

/// Re-express a symmetric function in another `SymFun` basis (`m`, `e`, `p`, `s`).
///
/// Implements the MAGMA coercion `A ! f` (change of basis, §146.2.2 / §146.5).
pub fn change_basis(sym: &SymFun, target: SymmetricFunctionBasis) -> SymFun {
    let cb = ClassicalBasis::from_symfun(target);
    let coeffs = coeff_map_in_basis(sym, cb);
    SymFun { basis: target, coeffs }
}

/// Expand a symmetric function into the power-sum hub.
pub fn to_powersum(sym: &SymFun) -> PMap {
    if sym.basis == SymmetricFunctionBasis::PowerSum {
        return sym
            .coeffs
            .iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(p, c)| (p.clone(), c.clone()))
            .collect();
    }
    coeff_map_in_basis(sym, ClassicalBasis::PowerSum)
}

/// Build a `SymFun` (in the given `SymFun` basis) from a power-sum combination.
pub fn from_powersum(pm: &PMap, target: SymmetricFunctionBasis) -> SymFun {
    let ps = SymFun {
        basis: SymmetricFunctionBasis::PowerSum,
        coeffs: pm.clone(),
    };
    change_basis(&ps, target)
}

// ---------------------------------------------------------------------------
// Ring product routed through the power-sum hub
// ---------------------------------------------------------------------------

/// The product of two symmetric functions, written (MAGMA convention) in the basis
/// of the *first* operand. Computed by expanding both operands in the power-sum
/// basis, merging partitions, and converting back.
pub fn multiply(a: &SymFun, b: &SymFun) -> SymFun {
    let pa = to_powersum(a);
    let pb = to_powersum(b);
    let prod = pmul(&pa, &pb);
    from_powersum(&prod, a.basis)
}

// ---------------------------------------------------------------------------
// Hall inner product (MAGMA §146.4.10)
// ---------------------------------------------------------------------------

/// The Hall inner product `<a, b>`, for which `<m_lambda, h_mu> = delta_{lambda mu}`,
/// the Schur functions are orthonormal, and `<p_lambda, p_mu> = z_lambda delta`.
pub fn hall_inner_product(a: &SymFun, b: &SymFun) -> Rational {
    let pa = to_powersum(a);
    let pb = to_powersum(b);
    let mut result = Rational::zero();
    for (part, ca) in &pa {
        if let Some(cb) = pb.get(part) {
            let z = centralizer_size(part);
            result = result + ca.clone() * cb.clone() * rati(z);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Omega / Frobenius involution (MAGMA §146.4.9)
// ---------------------------------------------------------------------------

/// The involution `omega` swapping `e_n <-> h_n` and `s_lambda <-> s_{lambda'}`.
/// On power sums `omega(p_k) = (-1)^{k-1} p_k`. The result is returned in the basis
/// of the input.
pub fn omega(sym: &SymFun) -> SymFun {
    let pm = to_powersum(sym);
    let mut out = PMap::new();
    for (part, c) in &pm {
        // sign = prod_i (-1)^{part_i - 1} = (-1)^{|part| - l(part)}
        let exp = part.sum() - part.length();
        let sign = if exp % 2 == 0 { 1 } else { -1 };
        padd(&mut out, part.clone(), c.clone() * rati(sign));
    }
    from_powersum(&out, sym.basis)
}

// ---------------------------------------------------------------------------
// Plethysm (MAGMA §146.4.5)
// ---------------------------------------------------------------------------

/// Apply the Adams operation `psi_k`: substitute `p_j -> p_{jk}` in a power-sum
/// combination. This is the power-sum plethysm `p_k[ - ]`, a ring homomorphism.
fn adams(pm: &PMap, k: usize) -> PMap {
    let mut out = PMap::new();
    for (part, c) in pm {
        let scaled: Vec<usize> = part.parts().iter().map(|&x| x * k).collect();
        padd(&mut out, Partition::new(scaled), c.clone());
    }
    out
}

/// Plethysm (composition) `f[g]`, returned in the basis of the *second* operand `g`
/// (MAGMA convention). `f[g] = sum_lambda f_p[lambda] prod_i psi_{lambda_i}(g)` where
/// `f = sum_lambda f_p[lambda] p_lambda` in the power-sum basis.
pub fn plethysm(f: &SymFun, g: &SymFun) -> SymFun {
    let fp = to_powersum(f);
    let gp = to_powersum(g);
    let mut result = PMap::new();
    for (lambda, cf) in &fp {
        let mut term = p_one();
        for &part in lambda.parts() {
            term = pmul(&term, &adams(&gp, part));
        }
        for (part, c) in term {
            padd(&mut result, part, c * cf.clone());
        }
    }
    from_powersum(&result, g.basis)
}

// ---------------------------------------------------------------------------
// Jacobi–Trudi determinants
// ---------------------------------------------------------------------------

fn heap_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut arr: Vec<usize> = (0..n).collect();
    let mut res = Vec::new();
    fn generate(k: usize, arr: &mut Vec<usize>, res: &mut Vec<Vec<usize>>) {
        if k == 1 {
            res.push(arr.clone());
            return;
        }
        for i in 0..k {
            generate(k - 1, arr, res);
            if k % 2 == 0 {
                arr.swap(i, k - 1);
            } else {
                arr.swap(0, k - 1);
            }
        }
    }
    generate(n, &mut arr, &mut res);
    res
}

fn perm_sign(p: &[usize]) -> i64 {
    let mut inv = 0usize;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            if p[i] > p[j] {
                inv += 1;
            }
        }
    }
    if inv % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Determinant of a matrix whose entries are power-sum combinations.
fn pdet(mat: &[Vec<PMap>]) -> PMap {
    let n = mat.len();
    if n == 0 {
        return p_one();
    }
    let mut total = PMap::new();
    for perm in heap_permutations(n) {
        let mut prod = p_one();
        for i in 0..n {
            prod = pmul(&prod, &mat[i][perm[i]]);
        }
        let s = perm_sign(&perm);
        for (part, c) in prod {
            padd(&mut total, part, c * rati(s));
        }
    }
    total
}

/// `h_i` (complete homogeneous) expanded in `p`, with `h_0 = 1` and `h_{<0} = 0`.
fn h_index_to_p(i: i64) -> PMap {
    if i < 0 {
        PMap::new()
    } else if i == 0 {
        p_one()
    } else {
        h_n_to_p(i as usize)
    }
}

/// `e_i` (elementary) expanded in `p`, with `e_0 = 1` and `e_{<0} = 0`.
fn e_index_to_p(i: i64) -> PMap {
    if i < 0 {
        PMap::new()
    } else if i == 0 {
        p_one()
    } else {
        e_n_to_p(i as usize)
    }
}

/// Schur function via the Jacobi–Trudi determinant `s_lambda = det(h_{lambda_i - i + j})`.
/// Returned in the Schur basis (so the result is `s_lambda` itself — a useful
/// self-check of the whole power-sum engine and the character table).
pub fn jacobi_trudi(lambda: &Partition) -> SymFun {
    let parts = lambda.parts();
    let l = parts.len();
    if l == 0 {
        let mut sf = SymFun::new(SymmetricFunctionBasis::Schur);
        sf.add_term(Partition::new(vec![]), Rational::one());
        return sf;
    }
    let mut mat: Vec<Vec<PMap>> = vec![vec![PMap::new(); l]; l];
    for i in 0..l {
        for j in 0..l {
            // 1-based index lambda_i - i + j
            let idx = parts[i] as i64 - (i as i64 + 1) + (j as i64 + 1);
            mat[i][j] = h_index_to_p(idx);
        }
    }
    let pm = pdet(&mat);
    from_powersum(&pm, SymmetricFunctionBasis::Schur)
}

/// Dual Jacobi–Trudi determinant `s_lambda = det(e_{lambda'_i - i + j})` using the
/// conjugate partition. Returned in the Schur basis.
pub fn dual_jacobi_trudi(lambda: &Partition) -> SymFun {
    let conj = lambda.conjugate();
    let parts = conj.parts();
    let l = parts.len();
    if l == 0 {
        let mut sf = SymFun::new(SymmetricFunctionBasis::Schur);
        sf.add_term(Partition::new(vec![]), Rational::one());
        return sf;
    }
    let mut mat: Vec<Vec<PMap>> = vec![vec![PMap::new(); l]; l];
    for i in 0..l {
        for j in 0..l {
            let idx = parts[i] as i64 - (i as i64 + 1) + (j as i64 + 1);
            mat[i][j] = e_index_to_p(idx);
        }
    }
    let pm = pdet(&mat);
    from_powersum(&pm, SymmetricFunctionBasis::Schur)
}

// ---------------------------------------------------------------------------
// Named MAGMA transition-matrix intrinsics (ch 146.5)
// ---------------------------------------------------------------------------

macro_rules! named_transition {
    ($name:ident, $from:expr, $to:expr) => {
        /// MAGMA ch 146.5 transition matrix over `Rational`.
        pub fn $name(n: usize) -> Matrix<Rational> {
            transition_matrix($from, $to, n)
        }
    };
}

named_transition!(schur_to_monomial_matrix, ClassicalBasis::Schur, ClassicalBasis::Monomial);
named_transition!(schur_to_homogeneous_matrix, ClassicalBasis::Schur, ClassicalBasis::Homogeneous);
named_transition!(schur_to_powersum_matrix, ClassicalBasis::Schur, ClassicalBasis::PowerSum);
named_transition!(schur_to_elementary_matrix, ClassicalBasis::Schur, ClassicalBasis::Elementary);

named_transition!(monomial_to_schur_matrix, ClassicalBasis::Monomial, ClassicalBasis::Schur);
named_transition!(monomial_to_homogeneous_matrix, ClassicalBasis::Monomial, ClassicalBasis::Homogeneous);
named_transition!(monomial_to_powersum_matrix, ClassicalBasis::Monomial, ClassicalBasis::PowerSum);
named_transition!(monomial_to_elementary_matrix, ClassicalBasis::Monomial, ClassicalBasis::Elementary);

named_transition!(homogeneous_to_schur_matrix, ClassicalBasis::Homogeneous, ClassicalBasis::Schur);
named_transition!(homogeneous_to_monomial_matrix, ClassicalBasis::Homogeneous, ClassicalBasis::Monomial);
named_transition!(homogeneous_to_powersum_matrix, ClassicalBasis::Homogeneous, ClassicalBasis::PowerSum);
named_transition!(homogeneous_to_elementary_matrix, ClassicalBasis::Homogeneous, ClassicalBasis::Elementary);

named_transition!(powersum_to_schur_matrix, ClassicalBasis::PowerSum, ClassicalBasis::Schur);
named_transition!(powersum_to_monomial_matrix, ClassicalBasis::PowerSum, ClassicalBasis::Monomial);
named_transition!(powersum_to_homogeneous_matrix, ClassicalBasis::PowerSum, ClassicalBasis::Homogeneous);
named_transition!(powersum_to_elementary_matrix, ClassicalBasis::PowerSum, ClassicalBasis::Elementary);

named_transition!(elementary_to_schur_matrix, ClassicalBasis::Elementary, ClassicalBasis::Schur);
named_transition!(elementary_to_monomial_matrix, ClassicalBasis::Elementary, ClassicalBasis::Monomial);
named_transition!(elementary_to_homogeneous_matrix, ClassicalBasis::Elementary, ClassicalBasis::Homogeneous);
named_transition!(elementary_to_powersum_matrix, ClassicalBasis::Elementary, ClassicalBasis::PowerSum);

// ---------------------------------------------------------------------------
// Constructors for classical basis elements as concrete SymFuns
// ---------------------------------------------------------------------------

/// The complete homogeneous symmetric function `h_lambda` as a concrete `SymFun`,
/// expressed in the Schur basis (since `SymFun` has no dedicated `h` variant).
pub fn complete_homogeneous(lambda: &Partition) -> SymFun {
    let pm = h_partition_to_p(lambda);
    from_powersum(&pm, SymmetricFunctionBasis::Schur)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(v: Vec<usize>) -> Partition {
        Partition::new(v)
    }

    #[test]
    fn test_mn_character_s3() {
        // Trivial rep chi^(3): all values 1.
        let triv = part(vec![3]);
        for rho in partitions(3) {
            assert_eq!(mn_character(&triv, &rho), 1, "trivial on {:?}", rho);
        }
        // Sign rep chi^(1,1,1): value = sign of permutation of that cycle type.
        let sign = part(vec![1, 1, 1]);
        assert_eq!(mn_character(&sign, &part(vec![3])), 1); // 3-cycle even
        assert_eq!(mn_character(&sign, &part(vec![2, 1])), -1); // transposition odd
        assert_eq!(mn_character(&sign, &part(vec![1, 1, 1])), 1); // identity
        // Standard rep chi^(2,1).
        let std = part(vec![2, 1]);
        assert_eq!(mn_character(&std, &part(vec![3])), -1);
        assert_eq!(mn_character(&std, &part(vec![2, 1])), 0);
        assert_eq!(mn_character(&std, &part(vec![1, 1, 1])), 2);
    }

    #[test]
    fn test_mn_dimension_is_stdtab_count() {
        // chi^lambda(1^n) = dimension = number of standard tableaux = partition.dimension().
        for n in 1..=6 {
            let ones = part(vec![1; n]);
            for lam in partitions(n) {
                let d = mn_character(&lam, &ones);
                assert_eq!(
                    d as usize,
                    lam.dimension(),
                    "dim mismatch for {:?}",
                    lam
                );
            }
        }
    }

    #[test]
    fn test_character_orthogonality_rows() {
        // Row orthogonality: sum_rho chi^lambda(rho) chi^mu(rho) / z_rho = delta.
        let n = 5;
        let ps = partitions(n);
        for lam in &ps {
            for mu in &ps {
                let mut s = Rational::zero();
                for rho in &ps {
                    let a = mn_character(lam, rho);
                    let b = mn_character(mu, rho);
                    let z = centralizer_size(rho);
                    s = s + rati(a) * rati(b) * rrecip(z);
                }
                let expected = if lam == mu { Rational::one() } else { Rational::zero() };
                assert_eq!(s, expected, "orthogonality {:?} {:?}", lam, mu);
            }
        }
    }

    #[test]
    fn test_schur_to_monomial_is_kostka() {
        // SchurToMonomial entries are the Kostka numbers K_{lambda,mu}. We validate
        // against hand-checked values and the defining structural properties
        // (K_{lambda,lambda} = 1; K_{lambda,mu} = 0 unless lambda dominates mu).
        // NOTE: the crate's independent `kostka::kostka_number` is buggy for some
        // inputs (e.g. it returns 2 for K_{(4,1),(3,2)} whose true value is 1), so
        // we do NOT compare against it here.

        // n = 3 fully by hand:  s_3 = m_3 + m_21 + m_111,
        //   s_21 = m_21 + 2 m_111,  s_111 = m_111.
        let ps3 = partitions(3);
        let idx = |p: &Partition| ps3.iter().position(|q| q == p).unwrap();
        let m3 = schur_to_monomial_matrix(3);
        let get = |lam: Vec<usize>, mu: Vec<usize>| {
            m3.get(idx(&part(lam)), idx(&part(mu))).unwrap().clone()
        };
        assert_eq!(get(vec![3], vec![3]), Rational::one());
        assert_eq!(get(vec![3], vec![2, 1]), Rational::one());
        assert_eq!(get(vec![3], vec![1, 1, 1]), Rational::one());
        assert_eq!(get(vec![2, 1], vec![2, 1]), Rational::one());
        assert_eq!(get(vec![2, 1], vec![1, 1, 1]), rati(2));
        assert_eq!(get(vec![2, 1], vec![3]), Rational::zero());
        assert_eq!(get(vec![1, 1, 1], vec![1, 1, 1]), Rational::one());

        // Structural properties for a range of degrees.
        for n in 1..=6 {
            let ps = partitions(n);
            let mat = schur_to_monomial_matrix(n);
            for (i, lam) in ps.iter().enumerate() {
                assert_eq!(*mat.get(i, i).unwrap(), Rational::one(), "diag {:?}", lam);
                for (j, mu) in ps.iter().enumerate() {
                    let entry = mat.get(i, j).unwrap().clone();
                    if lam != mu && !lam.dominates(mu) {
                        assert_eq!(entry, Rational::zero(), "K {:?} {:?} must vanish", lam, mu);
                    }
                    assert!(entry >= Rational::zero(), "K {:?} {:?} >= 0", lam, mu);
                }
            }
        }
    }

    #[test]
    fn test_homogeneous_to_monomial_symmetric() {
        // HomogeneousToMonomialMatrix is symmetric ([Mac95], H146E20).
        for n in 1..=6 {
            let mat = homogeneous_to_monomial_matrix(n);
            let len = partitions(n).len();
            for i in 0..len {
                for j in 0..len {
                    assert_eq!(mat.get(i, j).unwrap(), mat.get(j, i).unwrap());
                }
            }
        }
    }

    #[test]
    fn test_elementary_equals_homogeneous_omega_matrix() {
        // HomogeneousToElementaryMatrix(n) == ElementaryToHomogeneousMatrix(n)
        // (H146E21): both equal the involution matrix.
        for n in 1..=6 {
            let a = homogeneous_to_elementary_matrix(n);
            let b = elementary_to_homogeneous_matrix(n);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_schur_to_monomial_transpose_homogeneous_to_schur() {
        // SchurToMonomialMatrix(n) == transpose(HomogeneousToSchurMatrix(n)) (H146E19).
        for n in 1..=6 {
            let a = schur_to_monomial_matrix(n);
            let b = homogeneous_to_schur_matrix(n).transpose();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_powersum_to_schur_is_character_table() {
        // PowerSumToSchurMatrix entries are chi^lambda(rho): entry (rho, lambda).
        let n = 5;
        let ps = partitions(n);
        let mat = powersum_to_schur_matrix(n);
        for (i, rho) in ps.iter().enumerate() {
            for (j, lam) in ps.iter().enumerate() {
                let expected = rati(mn_character(lam, rho));
                assert_eq!(*mat.get(i, j).unwrap(), expected);
            }
        }
    }

    #[test]
    fn test_jacobi_trudi_recovers_schur() {
        for n in 1..=6 {
            for lam in partitions(n) {
                let jt = jacobi_trudi(&lam);
                assert_eq!(jt.basis, SymmetricFunctionBasis::Schur);
                assert_eq!(jt.coeff(&lam), Rational::one(), "s_{:?}", lam);
                // Exactly one nonzero term.
                assert_eq!(jt.support().len(), 1);
            }
        }
    }

    #[test]
    fn test_dual_jacobi_trudi_recovers_schur() {
        for n in 1..=5 {
            for lam in partitions(n) {
                let jt = dual_jacobi_trudi(&lam);
                assert_eq!(jt.coeff(&lam), Rational::one(), "dual s_{:?}", lam);
                assert_eq!(jt.support().len(), 1);
            }
        }
    }

    #[test]
    fn test_change_basis_roundtrip() {
        // s_{2,1} -> monomial -> schur returns s_{2,1}.
        let s = crate::basis::schur_function(part(vec![2, 1]));
        let m = change_basis(&s, SymmetricFunctionBasis::Monomial);
        assert_eq!(m.coeff(&part(vec![2, 1])), Rational::one()); // Kostka diagonal
        assert_eq!(m.coeff(&part(vec![1, 1, 1])), rati(2)); // K_{(2,1),(1^3)} = 2
        let back = change_basis(&m, SymmetricFunctionBasis::Schur);
        assert_eq!(back.coeff(&part(vec![2, 1])), Rational::one());
        assert_eq!(back.support().len(), 1);
    }

    #[test]
    fn test_hall_inner_product_schur_orthonormal() {
        let s21 = crate::basis::schur_function(part(vec![2, 1]));
        let s3 = crate::basis::schur_function(part(vec![3]));
        assert_eq!(hall_inner_product(&s21, &s21), Rational::one());
        assert_eq!(hall_inner_product(&s21, &s3), Rational::zero());
    }

    #[test]
    fn test_hall_m_h_duality() {
        // <m_lambda, h_mu> = delta.  h_mu built as a SymFun via complete_homogeneous.
        let n = 4;
        for lam in partitions(n) {
            let m = crate::basis::monomial_symmetric(lam.clone());
            for mu in partitions(n) {
                let h = complete_homogeneous(&mu);
                let ip = hall_inner_product(&m, &h);
                let expected = if lam == mu { Rational::one() } else { Rational::zero() };
                assert_eq!(ip, expected, "<m_{:?}, h_{:?}>", lam, mu);
            }
        }
    }

    #[test]
    fn test_omega_swaps_schur_conjugate() {
        // omega(s_{3,1}) = s_{2,1,1}.
        let s = crate::basis::schur_function(part(vec![3, 1]));
        let w = omega(&s);
        let conj = part(vec![3, 1]).conjugate();
        assert_eq!(w.coeff(&conj), Rational::one());
        assert_eq!(w.support().len(), 1);
    }

    #[test]
    fn test_omega_involutive() {
        let s = crate::basis::schur_function(part(vec![3, 1]));
        let ww = omega(&omega(&s));
        assert_eq!(ww.coeff(&part(vec![3, 1])), Rational::one());
        assert_eq!(ww.support().len(), 1);
    }

    #[test]
    fn test_multiply_schur_pieri() {
        // s_1 * s_1 = s_2 + s_{1,1} (Pieri).
        let s1 = crate::basis::schur_function(part(vec![1]));
        let prod = multiply(&s1, &s1);
        assert_eq!(prod.basis, SymmetricFunctionBasis::Schur);
        assert_eq!(prod.coeff(&part(vec![2])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![1, 1])), Rational::one());
        assert_eq!(prod.support().len(), 2);
    }

    #[test]
    fn test_multiply_lr_coefficient() {
        // s_{2,1} * s_{2,1} = s_{4,2}+s_{4,1,1}+s_{3,3}+2 s_{3,2,1}
        //   + s_{3,1,1,1}+s_{2,2,2}+s_{2,2,1,1}  (classic Littlewood-Richardson).
        let s21 = crate::basis::schur_function(part(vec![2, 1]));
        let prod = multiply(&s21, &s21);
        assert_eq!(prod.coeff(&part(vec![4, 2])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![4, 1, 1])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![3, 3])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![3, 2, 1])), rati(2));
        assert_eq!(prod.coeff(&part(vec![3, 1, 1, 1])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![2, 2, 2])), Rational::one());
        assert_eq!(prod.coeff(&part(vec![2, 2, 1, 1])), Rational::one());
        assert_eq!(prod.support().len(), 7);
    }

    #[test]
    fn test_plethysm_powersum() {
        // p_2[p_3] = p_6.
        let p2 = crate::basis::power_sum_symmetric(part(vec![2]));
        let p3 = crate::basis::power_sum_symmetric(part(vec![3]));
        let r = plethysm(&p2, &p3);
        assert_eq!(r.coeff(&part(vec![6])), Rational::one());
        assert_eq!(r.support().len(), 1);
    }

    #[test]
    fn test_plethysm_h2_of_h2_degree() {
        // h_2[h_2] is homogeneous of degree 4; in Schur basis it is s_4 + s_{2,2}.
        let h2 = complete_homogeneous(&part(vec![2]));
        let r = plethysm(&h2, &h2);
        let sr = change_basis(&r, SymmetricFunctionBasis::Schur);
        assert_eq!(sr.coeff(&part(vec![4])), Rational::one());
        assert_eq!(sr.coeff(&part(vec![2, 2])), Rational::one());
        assert_eq!(sr.support().len(), 2);
    }
}
