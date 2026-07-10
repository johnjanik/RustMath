//! Lattices: a first-class `Lattice` parent, arbitrary-precision Gram–Schmidt /
//! LLL, and Fincke–Pohst short-vector enumeration (MAGMA ch. 30).
//!
//! MAGMA source: Handbook chapter 30 (Lattices), §30.2 (basis / Gram matrix),
//! §30.4 (`Rank`, `Determinant`, `GramMatrix`, `Basis`), §30.7.1 (`LLL`) and
//! §30.8 (`ShortVectors`, `ShortestVectors` via Fincke–Pohst `[FP83]`).
//!
//! ## Relation to the existing `lll.rs`
//!
//! The crate already ships a pure-integer LLL whose Gram–Schmidt is steered in
//! `f64` (`lll.rs`). This module is **purely additive** and sits beside it: the
//! reduction here computes the Gram–Schmidt orthogonalisation in
//! [`rustmath_reals::BigFloat`] (an arbitrary-precision [`RealField`]), which is
//! far better conditioned than `f64` for near-degenerate or large-entry bases.
//! As in `lll.rs`, every basis update is an **exact integer** operation, so the
//! returned transform is exactly unimodular and `reduced = U · basis` holds
//! exactly regardless of floating rounding — the real field only decides *how*
//! the reduction proceeds, never the arithmetic of the result.

use rustmath_core::analytic::RealField;
use rustmath_core::ordering::OrderedRing;
use rustmath_core::parent::{Parent, ParentWithGenerators};
use rustmath_core::EuclideanDomain;
use rustmath_core::MathError;
use rustmath_core::Result;
use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;
use std::collections::BTreeSet;

use crate::Matrix;

/// Default working precision (bits) for the real-field Gram–Schmidt / LLL.
pub const LATTICE_PRECISION: u64 = 128;

// ------------------------------------------------------------------------- //
// Generic RealField helpers                                                  //
// ------------------------------------------------------------------------- //

/// Convert an [`Integer`] to any [`RealField`] element at `prec` bits, by
/// base-2³² Horner evaluation. Every limb is exactly representable in an `f64`,
/// so the only inexactness is the rounding to `prec` bits that any float
/// constructor performs.
fn rf_from_integer<F: RealField>(n: &Integer, prec: u64) -> F {
    if n.bit_length() <= 52 {
        // |n| < 2^53: exactly representable in f64.
        return F::from_f64(n.to_i64() as f64, prec);
    }
    let base_int = Integer::from(1i64 << 32);
    let mut m = n.abs();
    let mut limbs: Vec<f64> = Vec::new();
    while !m.is_zero() {
        let (q, r) = m.div_rem(&base_int).expect("2^32 is non-zero");
        limbs.push(r.to_i64() as f64); // r ∈ [0, 2^32): exact in f64
        m = q;
    }
    let base = F::from_f64(4294967296.0, prec);
    let mut acc = F::from_f64(0.0, prec);
    for &d in limbs.iter().rev() {
        acc = acc * base.clone() + F::from_f64(d, prec);
    }
    if n.signum() < 0 {
        -acc
    } else {
        acc
    }
}

/// `BigFloat` shorthand (exact constructor; used by the enumeration code).
fn bf(n: &Integer, prec: u64) -> BigFloat {
    BigFloat::from_integer(n, prec)
}

/// Nearest integer to a real-field value, seeded from its `f64` value and
/// corrected by exact comparison in `F`. The correction loop is capped: an
/// unconverged seed only degrades reduction quality, never the exactness of
/// the (integer) basis update that consumes it.
fn round_to_int<F: RealField>(x: &F, prec: u64) -> Integer {
    let approx = x.to_f64();
    let mut g = if approx.is_finite() && approx.abs() < 9.0e18 {
        Integer::from(approx.round() as i64)
    } else {
        Integer::zero()
    };
    let half = F::from_f64(0.5, prec);
    let neg_half = -half.clone();
    for _ in 0..128 {
        let diff = x.clone() - rf_from_integer::<F>(&g, prec); // x - g
        if diff > half {
            g = g + Integer::one();
        } else if diff < neg_half {
            g = g - Integer::one();
        } else {
            break;
        }
    }
    g
}

fn floor_to_int<F: RealField>(x: &F, prec: u64) -> Integer {
    let r = round_to_int(x, prec);
    if rf_from_integer::<F>(&r, prec) > *x {
        r - Integer::one()
    } else {
        r
    }
}

fn ceil_to_int<F: RealField>(x: &F, prec: u64) -> Integer {
    let r = round_to_int(x, prec);
    if rf_from_integer::<F>(&r, prec) < *x {
        r + Integer::one()
    } else {
        r
    }
}

/// Exact integer dot product.
fn idot(a: &[Integer], b: &[Integer]) -> Integer {
    let mut s = Integer::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        s = s + x.clone() * y.clone();
    }
    s
}

/// Gram–Schmidt orthogonalisation of integer rows, carried in any
/// [`RealField`] `F` (both `BigFloat` and `RealMPFR` work) at `prec` bits.
/// Returns `(mu, bstar_norm2)` with `mu[i][j]` the GSO coefficients
/// (`mu[i][i] = 1`) and `bnorm[i] = ‖b_i*‖²`.
///
/// For the *exact* rational orthogonalisation, see [`gram_schmidt_exact`].
pub fn gram_schmidt_real<F: RealField>(
    b: &[Vec<Integer>],
    prec: u64,
) -> (Vec<Vec<F>>, Vec<F>) {
    let n = b.len();
    let zero = F::from_f64(0.0, prec);
    let mut mu = vec![vec![zero.clone(); n]; n];
    let mut bnorm = vec![zero.clone(); n];
    let mut bstar: Vec<Vec<F>> = Vec::with_capacity(n);

    for i in 0..n {
        let bi: Vec<F> = b[i].iter().map(|x| rf_from_integer(x, prec)).collect();
        let mut v = bi.clone();
        for j in 0..i {
            // dot(b_i, b_j*)
            let mut dotp = zero.clone();
            for d in 0..v.len() {
                dotp = dotp + bi[d].clone() * bstar[j][d].clone();
            }
            let m = if bnorm[j].sign() != 0 {
                dotp / bnorm[j].clone()
            } else {
                zero.clone()
            };
            mu[i][j] = m.clone();
            for d in 0..v.len() {
                v[d] = v[d].clone() - m.clone() * bstar[j][d].clone();
            }
        }
        mu[i][i] = F::from_f64(1.0, prec);
        let mut nrm = zero.clone();
        for d in 0..v.len() {
            nrm = nrm + v[d].clone() * v[d].clone();
        }
        bnorm[i] = nrm;
        bstar.push(v);
    }
    (mu, bnorm)
}

/// Gram–Schmidt orthogonalisation of integer rows in `BigFloat` (the concrete
/// instantiation used by the [`Lattice`] parent).
fn gram_schmidt_bf(b: &[Vec<Integer>], prec: u64) -> (Vec<Vec<BigFloat>>, Vec<BigFloat>) {
    gram_schmidt_real::<BigFloat>(b, prec)
}

/// **Exact** Gram–Schmidt orthogonalisation over ℚ.
///
/// Returns `(bstar, mu)` where `bstar[i]` are the orthogonalised rows and `mu`
/// is the lower-triangular coefficient matrix (`mu[i][i] = 1`,
/// `mu[i][j] = ⟨bᵢ, bⱼ*⟩ / ‖bⱼ*‖²` for `j < i`), all computed exactly in
/// rational arithmetic — no rounding of any kind.
///
/// Linearly dependent rows are permitted: a dependent row orthogonalises to
/// the zero vector, and later projections simply skip it (its `mu` column is
/// zero below the diagonal).
///
/// Errors if the rows do not all have the same length.
pub fn gram_schmidt_exact(
    b: &[Vec<Rational>],
) -> Result<(Vec<Vec<Rational>>, Vec<Vec<Rational>>)> {
    let n = b.len();
    if n > 0 {
        let w = b[0].len();
        if b.iter().any(|r| r.len() != w) {
            return Err(MathError::InvalidArgument(
                "gram_schmidt_exact: all rows must have the same length".to_string(),
            ));
        }
    }
    let zero = Rational::from_integer(0);
    let one = Rational::from_integer(1);
    let mut mu = vec![vec![zero.clone(); n]; n];
    let mut bstar: Vec<Vec<Rational>> = Vec::with_capacity(n);

    for i in 0..n {
        let mut v = b[i].clone();
        for j in 0..i {
            let mut dotp = zero.clone();
            let mut nrm = zero.clone();
            for d in 0..v.len() {
                dotp = dotp + b[i][d].clone() * bstar[j][d].clone();
                nrm = nrm + bstar[j][d].clone() * bstar[j][d].clone();
            }
            let m = if nrm.is_zero() {
                zero.clone()
            } else {
                dotp / nrm
            };
            mu[i][j] = m.clone();
            for d in 0..v.len() {
                v[d] = v[d].clone() - m.clone() * bstar[j][d].clone();
            }
        }
        mu[i][i] = one.clone();
        bstar.push(v);
    }
    Ok((bstar, mu))
}

/// **Exact** LLL-reducedness certificate over ℚ, for an integer basis.
///
/// Checks, in exact rational arithmetic (via [`gram_schmidt_exact`]):
///
/// 1. size reduction: `|μ_{i,j}| ≤ 1/2` for all `j < i`, and
/// 2. the Lovász condition
///    `‖b_k*‖² ≥ (δ − μ_{k,k-1}²)·‖b_{k-1}*‖²` with `δ = delta_num/delta_den`.
///
/// This is a rigorous certificate: no floating point is involved, so a `true`
/// answer *proves* the basis is LLL-reduced for the given `δ`.
pub fn lll_is_reduced_exact(
    basis: &[Vec<Integer>],
    delta_num: i64,
    delta_den: i64,
) -> Result<bool> {
    let rows: Vec<Vec<Rational>> = basis
        .iter()
        .map(|r| r.iter().map(|x| Rational::from_integer(x.clone())).collect())
        .collect();
    let (bstar, mu) = gram_schmidt_exact(&rows)?;
    let n = rows.len();
    let half = Rational::new(Integer::one(), Integer::from(2))?;
    let delta = Rational::new(Integer::from(delta_num), Integer::from(delta_den))?;

    let norm2 = |v: &[Rational]| -> Rational {
        let mut s = Rational::from_integer(0);
        for x in v {
            s = s + x.clone() * x.clone();
        }
        s
    };

    for i in 1..n {
        for j in 0..i {
            if mu[i][j].abs() > half {
                return Ok(false);
            }
        }
    }
    for k in 1..n {
        let lhs = norm2(&bstar[k]);
        let m = mu[k][k - 1].clone();
        let rhs = (delta.clone() - m.clone() * m) * norm2(&bstar[k - 1]);
        if lhs < rhs {
            return Ok(false);
        }
    }
    Ok(true)
}

/// LLL-reduce the integer lattice spanned by `basis` (rows), with the
/// orthogonalisation carried in any [`RealField`] `F` at `prec` bits — both
/// `rustmath_reals::BigFloat` and `rustmath_reals::RealMPFR` work. Returns
/// `(reduced, u)` with `reduced = u · basis` **exactly** and `u` unimodular.
///
/// The Lovász parameter is `δ = delta_num / delta_den`; the standard default
/// is `δ = 3/4` (pass `3, 4`), and `1/4 < δ < 1` is required for the classic
/// complexity/quality guarantees.
///
/// As in [`crate::lll::lll_reduce`], every basis update is an exact integer
/// operation, so the returned transform is exactly unimodular regardless of
/// floating rounding — the real field only steers *how* the reduction
/// proceeds. Output quality can be certified exactly with
/// [`lll_is_reduced_exact`].
pub fn lll_reduce_real<F: RealField>(
    basis: &[Vec<Integer>],
    delta_num: i64,
    delta_den: i64,
    prec: u64,
) -> (Vec<Vec<Integer>>, Vec<Vec<Integer>>) {
    let n = basis.len();
    let mut b: Vec<Vec<Integer>> = basis.to_vec();
    let mut u: Vec<Vec<Integer>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { Integer::one() } else { Integer::zero() })
                .collect()
        })
        .collect();
    if n <= 1 {
        return (b, u);
    }

    let delta =
        F::from_f64(delta_num as f64, prec) / F::from_f64(delta_den as f64, prec);
    let half = F::from_f64(0.5, prec);

    let (mut mu, mut bnorm) = gram_schmidt_real::<F>(&b, prec);
    let cap = 1000 * n * n + 1000;
    let mut iters = 0usize;
    let mut k = 1usize;
    while k < n {
        iters += 1;
        if iters > cap {
            break;
        }
        // size-reduce b[k] against b[k-1..=0]
        let mut changed = false;
        for j in (0..k).rev() {
            if mu[k][j].abs() > half {
                let r = round_to_int(&mu[k][j], prec);
                if !r.is_zero() {
                    let cols = b[k].len();
                    for d in 0..cols {
                        b[k][d] = b[k][d].clone() - r.clone() * b[j][d].clone();
                    }
                    for d in 0..n {
                        u[k][d] = u[k][d].clone() - r.clone() * u[j][d].clone();
                    }
                    changed = true;
                }
            }
        }
        if changed {
            let g = gram_schmidt_real::<F>(&b, prec);
            mu = g.0;
            bnorm = g.1;
        }
        // Lovász condition: ‖b_k*‖² ≥ (δ − μ_{k,k-1}²) ‖b_{k-1}*‖²
        let lhs = bnorm[k].clone();
        let mkk = mu[k][k - 1].clone();
        let rhs = (delta.clone() - mkk.clone() * mkk) * bnorm[k - 1].clone();
        if lhs >= rhs {
            k += 1;
        } else {
            b.swap(k, k - 1);
            u.swap(k, k - 1);
            let g = gram_schmidt_real::<F>(&b, prec);
            mu = g.0;
            bnorm = g.1;
            k = if k > 1 { k - 1 } else { 1 };
        }
    }
    (b, u)
}

/// LLL-reduce with the orthogonalisation carried in `prec`-bit [`BigFloat`]
/// (the concrete instantiation of [`lll_reduce_real`]). Returns `(reduced, u)`
/// with `reduced = u · basis` exactly and `u` unimodular.
///
/// This is the real-field analogue of [`crate::lll::lll_reduce`]; see the module
/// docs for the exactness guarantee.
pub fn lll_reduce_rf(
    basis: &[Vec<Integer>],
    delta_num: i64,
    delta_den: i64,
    prec: u64,
) -> (Vec<Vec<Integer>>, Vec<Vec<Integer>>) {
    lll_reduce_real::<BigFloat>(basis, delta_num, delta_den, prec)
}

// ------------------------------------------------------------------------- //
// Lattice parent                                                            //
// ------------------------------------------------------------------------- //

/// A full-rank-or-lower integer lattice `L = ⟨b₁, …, b_m⟩ ⊆ ℤⁿ` with the
/// standard Euclidean inner product. The rows of the basis are assumed
/// independent (as for MAGMA's `LatticeWithBasis`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lattice {
    /// Basis vectors (rows), each of ambient length `n`.
    basis: Vec<Vec<Integer>>,
    /// Ambient dimension `n`.
    ambient: usize,
    /// Working precision for real-field reductions.
    precision: u64,
}

impl Lattice {
    /// Build a lattice from independent basis rows (each of equal length).
    pub fn with_basis(basis: Vec<Vec<Integer>>) -> Result<Self> {
        if basis.is_empty() {
            return Err(MathError::InvalidArgument(
                "lattice basis cannot be empty".to_string(),
            ));
        }
        let ambient = basis[0].len();
        if ambient == 0 || basis.iter().any(|r| r.len() != ambient) {
            return Err(MathError::InvalidArgument(
                "all basis rows must share a positive ambient length".to_string(),
            ));
        }
        Ok(Lattice {
            basis,
            ambient,
            precision: LATTICE_PRECISION,
        })
    }

    /// Build a lattice from a generating set, LLL-reducing it to a basis
    /// (MAGMA `Lattice(X)`). Zero rows produced by reduction are dropped.
    pub fn from_generators(gens: Vec<Vec<Integer>>) -> Result<Self> {
        let lat = Lattice::with_basis(gens)?;
        let (reduced, _u) = lll_reduce_rf(&lat.basis, 3, 4, lat.precision);
        let nonzero: Vec<Vec<Integer>> = reduced
            .into_iter()
            .filter(|r| r.iter().any(|x| !x.is_zero()))
            .collect();
        Lattice::with_basis(nonzero)
    }

    /// Override the working precision (bits) for real-field reductions.
    pub fn with_precision(mut self, precision: u64) -> Self {
        self.precision = precision.max(1);
        self
    }

    /// The rank (number of basis vectors) of the lattice.
    pub fn rank(&self) -> usize {
        self.basis.len()
    }

    /// The ambient dimension `n` (`Degree`/`OverDimension` in MAGMA).
    pub fn ambient_dimension(&self) -> usize {
        self.ambient
    }

    /// The basis rows (MAGMA `Basis` / `BasisMatrix`).
    pub fn basis(&self) -> &[Vec<Integer>] {
        &self.basis
    }

    /// The `m × m` Gram matrix `F = B Bᵀ` under the standard inner product
    /// (MAGMA `GramMatrix`).
    pub fn gram_matrix(&self) -> Matrix<Integer> {
        let m = self.rank();
        let mut data = vec![Integer::zero(); m * m];
        for i in 0..m {
            for j in 0..m {
                data[i * m + j] = idot(&self.basis[i], &self.basis[j]);
            }
        }
        Matrix::from_vec(m, m, data).expect("gram shape is consistent")
    }

    /// The lattice determinant *squared*, `det(F) = det(B Bᵀ)` (exact). The true
    /// determinant is its square root; kept exact here for integrality.
    pub fn determinant_squared(&self) -> Result<Integer> {
        self.gram_matrix().det()
    }

    /// The LLL-reduced lattice (real-field steered, `δ = 3/4`), together with the
    /// unimodular transform `U` with `reduced = U · basis`.
    pub fn lll_reduced(&self) -> (Lattice, Vec<Vec<Integer>>) {
        let (reduced, u) = lll_reduce_rf(&self.basis, 3, 4, self.precision);
        let lat = Lattice {
            basis: reduced,
            ambient: self.ambient,
            precision: self.precision,
        };
        (lat, u)
    }

    /// Realise an integer coordinate vector as the lattice element
    /// `Σ coords[i] · bᵢ`.
    pub fn coordinates_to_vector(&self, coords: &[Integer]) -> Result<Vec<Integer>> {
        if coords.len() != self.rank() {
            return Err(MathError::InvalidArgument(
                "coordinate length must equal the lattice rank".to_string(),
            ));
        }
        let mut v = vec![Integer::zero(); self.ambient];
        for (c, row) in coords.iter().zip(self.basis.iter()) {
            for d in 0..self.ambient {
                v[d] = v[d].clone() + c.clone() * row[d].clone();
            }
        }
        Ok(v)
    }

    /// All non-zero lattice vectors `v` with `0 < ‖v‖² ≤ bound`, one per `±`
    /// pair, via Fincke–Pohst enumeration (MAGMA `ShortVectors`, `[FP83]`).
    ///
    /// Returns `(vector, squared_norm)` pairs. Enumeration is capped to guard
    /// against pathological inputs; a hit-count near the cap is reported by an
    /// empty result only if truly nothing was found.
    pub fn short_vectors(&self, bound: &Integer) -> Vec<(Vec<Integer>, Integer)> {
        if bound.signum() <= 0 {
            return Vec::new();
        }
        let prec = self.precision;
        let (mu, bnorm) = gram_schmidt_bf(&self.basis, prec);
        let n = self.rank();
        let cbound = bf(bound, prec);

        // Sign-normalised coefficient vectors already emitted.
        let mut seen: BTreeSet<Vec<Integer>> = BTreeSet::new();
        let mut out: Vec<(Vec<Integer>, Integer)> = Vec::new();
        let mut x = vec![Integer::zero(); n];
        let mut nodes: u64 = 0;
        const NODE_CAP: u64 = 2_000_000;

        self.fp_enumerate(
            n as isize - 1,
            &mut x,
            &mu,
            &bnorm,
            &cbound,
            &BigFloat::from_integer(&Integer::zero(), prec),
            bound,
            prec,
            &mut seen,
            &mut out,
            &mut nodes,
            NODE_CAP,
        );
        out
    }

    /// The non-zero lattice vectors of minimal norm (MAGMA `ShortestVectors`).
    /// Uses the shortest LLL basis vector as the enumeration radius.
    pub fn shortest_vectors(&self) -> Vec<(Vec<Integer>, Integer)> {
        let (reduced, _u) = self.lll_reduced();
        let bound = reduced
            .basis
            .iter()
            .map(|r| idot(r, r))
            .filter(|nrm| nrm.signum() > 0)
            .min();
        let bound = match bound {
            Some(b) => b,
            None => return Vec::new(),
        };
        let mut all = self.short_vectors(&bound);
        // keep only the minimal-norm ones
        let min = all.iter().map(|(_, n)| n.clone()).min();
        if let Some(minv) = min {
            all.retain(|(_, nrm)| *nrm == minv);
        }
        all
    }

    #[allow(clippy::too_many_arguments)]
    fn fp_enumerate(
        &self,
        k: isize,
        x: &mut Vec<Integer>,
        mu: &[Vec<BigFloat>],
        bnorm: &[BigFloat],
        cbound: &BigFloat,
        rho_above: &BigFloat,
        bound_int: &Integer,
        prec: u64,
        seen: &mut BTreeSet<Vec<Integer>>,
        out: &mut Vec<(Vec<Integer>, Integer)>,
        nodes: &mut u64,
        node_cap: u64,
    ) {
        if *nodes > node_cap {
            return;
        }
        *nodes += 1;

        if k < 0 {
            // Reached a full coordinate vector. Skip zero.
            if x.iter().all(|c| c.is_zero()) {
                return;
            }
            // Sign-normalise: make the first non-zero coefficient positive.
            let mut coeff = x.clone();
            if let Some(first) = coeff.iter().find(|c| !c.is_zero()) {
                if first.signum() < 0 {
                    for c in coeff.iter_mut() {
                        *c = -c.clone();
                    }
                }
            }
            if !seen.insert(coeff.clone()) {
                return;
            }
            if let Ok(v) = self.coordinates_to_vector(&coeff) {
                let nrm = idot(&v, &v);
                if nrm.signum() > 0 && nrm <= *bound_int {
                    out.push((v, nrm));
                }
            }
            return;
        }

        let ki = k as usize;
        // budget = C - rho_above ; if negative, prune.
        let budget = cbound.clone() - rho_above.clone();
        if budget.sign() < 0 {
            return;
        }
        // y_k = Σ_{j>k} x_j μ_{j,k}
        let mut y = BigFloat::from_integer(&Integer::zero(), prec);
        for j in (ki + 1)..x.len() {
            if !x[j].is_zero() {
                y = y + bf(&x[j], prec) * mu[j][ki].clone();
            }
        }
        // |x_k + y| ≤ sqrt(budget / B_k)
        if bnorm[ki].sign() <= 0 {
            return;
        }
        let t = (budget / bnorm[ki].clone()).sqrt();
        let center = -y.clone();
        let lo = ceil_to_int(&(center.clone() - t.clone()), prec);
        let hi = floor_to_int(&(center + t), prec);

        let mut xk = lo;
        while xk <= hi {
            x[ki] = xk.clone();
            // term = (x_k + y)² · B_k
            let s = bf(&xk, prec) + y.clone();
            let term = s.clone() * s * bnorm[ki].clone();
            let rho_new = rho_above.clone() + term;
            if rho_new <= *cbound {
                self.fp_enumerate(
                    k - 1,
                    x,
                    mu,
                    bnorm,
                    cbound,
                    &rho_new,
                    bound_int,
                    prec,
                    seen,
                    out,
                    nodes,
                    node_cap,
                );
            }
            xk = xk + Integer::one();
            if *nodes > node_cap {
                break;
            }
        }
        x[ki] = Integer::zero();
    }
}

impl Parent for Lattice {
    type Element = Vec<Integer>;

    fn contains(&self, element: &Self::Element) -> bool {
        if element.len() != self.ambient {
            return false;
        }
        // v ∈ L iff v = Σ aᵢ bᵢ with aᵢ ∈ ℤ. Solve Bᵀ a = v over ℚ (unique since
        // the basis rows are independent), then require the solution integral.
        match self.solve_coordinates(element) {
            Some(coords) => coords.iter().all(|c| c.is_integer()),
            None => false,
        }
    }

    fn zero(&self) -> Option<Self::Element> {
        Some(vec![Integer::zero(); self.ambient])
    }

    fn cardinality(&self) -> Option<usize> {
        None // infinite (for a positive-rank lattice)
    }

    fn name(&self) -> String {
        format!(
            "Lattice of rank {} in Z^{}",
            self.rank(),
            self.ambient_dimension()
        )
    }
}

impl ParentWithGenerators for Lattice {
    fn generators(&self) -> Vec<Self::Element> {
        self.basis.clone()
    }
}

impl Lattice {
    /// Solve `Bᵀ a = v` over ℚ for the (unique) coordinate vector, or `None` if
    /// `v` is not in the ℚ-span of the basis. Straightforward exact Gaussian
    /// elimination on an `ambient × rank` system with independent columns.
    fn solve_coordinates(&self, v: &[Integer]) -> Option<Vec<Rational>> {
        let m = self.ambient; // equations
        let k = self.rank(); // unknowns
        // Augmented matrix rows: for each ambient coordinate d, the row is
        // [ b_0[d], b_1[d], ..., b_{k-1}[d] | v[d] ].
        let mut a: Vec<Vec<Rational>> = Vec::with_capacity(m);
        for d in 0..m {
            let mut row = Vec::with_capacity(k + 1);
            for i in 0..k {
                row.push(Rational::from_integer(self.basis[i][d].clone()));
            }
            row.push(Rational::from_integer(v[d].clone()));
            a.push(row);
        }

        // Forward elimination to reduced row echelon over the k unknowns.
        let mut pivot_row = 0usize;
        let mut where_pivot = vec![usize::MAX; k];
        for col in 0..k {
            // find a pivot at or below pivot_row
            let mut sel = None;
            for r in pivot_row..m {
                if !a[r][col].is_zero() {
                    sel = Some(r);
                    break;
                }
            }
            let sel = match sel {
                Some(r) => r,
                None => continue,
            };
            a.swap(pivot_row, sel);
            // normalise pivot row
            let inv = a[pivot_row][col].clone();
            for c in col..=k {
                a[pivot_row][c] = a[pivot_row][c].clone() / inv.clone();
            }
            // eliminate the column from all other rows
            for r in 0..m {
                if r != pivot_row && !a[r][col].is_zero() {
                    let factor = a[r][col].clone();
                    for c in col..=k {
                        a[r][c] = a[r][c].clone() - factor.clone() * a[pivot_row][c].clone();
                    }
                }
            }
            where_pivot[col] = pivot_row;
            pivot_row += 1;
        }

        // Consistency: any all-zero-on-unknowns row must have zero rhs.
        for r in 0..m {
            let lhs_zero = (0..k).all(|c| a[r][c].is_zero());
            if lhs_zero && !a[r][k].is_zero() {
                return None;
            }
        }

        // Read off the (unique) solution; independent columns ⇒ every column has
        // a pivot.
        let mut sol = vec![Rational::from_integer(0); k];
        for (col, &pr) in where_pivot.iter().enumerate() {
            if pr == usize::MAX {
                // free column ⇒ basis columns not independent; bail conservatively
                return None;
            }
            sol[col] = a[pr][k].clone();
        }
        Some(sol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[i64]) -> Vec<Integer> {
        xs.iter().map(|&x| Integer::from(x)).collect()
    }
    fn norm2(x: &[Integer]) -> Integer {
        idot(x, x)
    }

    #[test]
    fn round_helpers_are_correct() {
        let prec = 64;
        let x = BigFloat::from_rational(
            &Rational::new(Integer::from(7), Integer::from(2)).unwrap(),
            prec,
        ); // 3.5
        // round(3.5) -> 4 (ties up via > half then <= half boundary handling)
        assert_eq!(round_to_int(&x, prec), Integer::from(4));
        assert_eq!(floor_to_int(&x, prec), Integer::from(3));
        assert_eq!(ceil_to_int(&x, prec), Integer::from(4));

        let y = BigFloat::from_rational(
            &Rational::new(Integer::from(-7), Integer::from(2)).unwrap(),
            prec,
        ); // -3.5
        assert_eq!(floor_to_int(&y, prec), Integer::from(-4));
        assert_eq!(ceil_to_int(&y, prec), Integer::from(-3));
    }

    #[test]
    fn lll_rf_reduces_and_transforms_exactly() {
        // Cohen's classic 3×3 example.
        let basis = vec![v(&[1, 0, 0]), v(&[0, 1, 0]), v(&[3, 4, 5])];
        let (red, u) = lll_reduce_rf(&basis, 3, 4, 128);
        // reduced = u · basis, exactly
        for i in 0..3 {
            for d in 0..3 {
                let mut acc = Integer::zero();
                for j in 0..3 {
                    acc = acc + u[i][j].clone() * basis[j][d].clone();
                }
                assert_eq!(acc, red[i][d]);
            }
        }
        // The Gram determinant (a lattice invariant) is preserved.
        let g0 = Lattice::with_basis(basis).unwrap().determinant_squared().unwrap();
        let g1 = Lattice::with_basis(red).unwrap().determinant_squared().unwrap();
        assert_eq!(g0, g1);
    }

    #[test]
    fn lll_rf_shortens_a_bad_basis() {
        let basis = vec![v(&[100, 1]), v(&[101, 1])];
        let (red, _) = lll_reduce_rf(&basis, 3, 4, 128);
        let min_red = red.iter().map(|x| norm2(x)).min().unwrap();
        assert!(min_red < Integer::from(100));
    }

    #[test]
    fn lll_generic_wikipedia_example_bigfloat_and_mpfr() {
        // The worked example from the Wikipedia LLL article. Expected reduced
        // basis independently verified with SymPy:
        //   DomainMatrix([[1,1,1],[-1,0,2],[3,5,6]], ZZ).lll(delta=QQ(3,4))
        //     == [[0,1,0],[1,0,1],[-1,0,2]].
        let basis = vec![v(&[1, 1, 1]), v(&[-1, 0, 2]), v(&[3, 5, 6])];
        let expected = vec![v(&[0, 1, 0]), v(&[1, 0, 1]), v(&[-1, 0, 2])];

        let (red_bf, u_bf) = lll_reduce_real::<BigFloat>(&basis, 3, 4, 128);
        let (red_mp, u_mp) =
            lll_reduce_real::<rustmath_reals::RealMPFR>(&basis, 3, 4, 128);
        assert_eq!(red_bf, expected, "BigFloat-steered LLL");
        assert_eq!(red_mp, expected, "RealMPFR-steered LLL");

        // reduced = u · basis holds exactly for both backends
        for (red, u) in [(&red_bf, &u_bf), (&red_mp, &u_mp)] {
            for i in 0..3 {
                for d in 0..3 {
                    let mut acc = Integer::zero();
                    for j in 0..3 {
                        acc = acc + u[i][j].clone() * basis[j][d].clone();
                    }
                    assert_eq!(acc, red[i][d]);
                }
            }
        }

        // Exact rational certificate: output reduced, input not.
        assert!(lll_is_reduced_exact(&red_bf, 3, 4).unwrap());
        assert!(!lll_is_reduced_exact(&basis, 3, 4).unwrap());
    }

    #[test]
    fn gram_schmidt_exact_matches_sympy() {
        // SymPy: GramSchmidt([(3,1), (2,2)]) == [(3,1), (-2/5, 6/5)],
        // with mu_{1,0} = <(2,2),(3,1)>/||(3,1)||² = 8/10 = 4/5.
        let rows = vec![
            vec![Rational::from_integer(3), Rational::from_integer(1)],
            vec![Rational::from_integer(2), Rational::from_integer(2)],
        ];
        let (bstar, mu) = gram_schmidt_exact(&rows).unwrap();
        assert_eq!(bstar[0], rows[0]);
        assert_eq!(
            bstar[1],
            vec![Rational::new(-2, 5).unwrap(), Rational::new(6, 5).unwrap()]
        );
        assert_eq!(mu[1][0], Rational::new(4, 5).unwrap());

        // Orthogonality is exact, not approximate.
        let dot = bstar[0][0].clone() * bstar[1][0].clone()
            + bstar[0][1].clone() * bstar[1][1].clone();
        assert!(dot.is_zero());
    }

    #[test]
    fn gram_schmidt_exact_handles_dependent_rows() {
        let rows = vec![
            vec![Rational::from_integer(1), Rational::from_integer(2)],
            vec![Rational::from_integer(2), Rational::from_integer(4)],
        ];
        let (bstar, _mu) = gram_schmidt_exact(&rows).unwrap();
        assert!(bstar[1].iter().all(|x| x.is_zero()));

        // ragged input errors
        let bad = vec![vec![Rational::from_integer(1)], vec![]];
        assert!(gram_schmidt_exact(&bad).is_err());
    }

    #[test]
    fn gram_schmidt_real_agrees_with_exact_gs() {
        // The generic real-field GS (BigFloat and RealMPFR) must agree with the
        // exact rational GS to well within the 128-bit working precision.
        let basis = vec![v(&[1, 1, 1]), v(&[-1, 0, 2]), v(&[3, 5, 6])];
        let rows: Vec<Vec<Rational>> = basis
            .iter()
            .map(|r| r.iter().map(|x| Rational::from_integer(x.clone())).collect())
            .collect();
        let (bstar, mu_q) = gram_schmidt_exact(&rows).unwrap();
        let exact_norm2 = |i: usize| -> f64 {
            let mut s = Rational::from_integer(0);
            for x in &bstar[i] {
                s = s + x.clone() * x.clone();
            }
            s.to_f64().unwrap()
        };

        let (mu_bf, bnorm_bf) = gram_schmidt_real::<BigFloat>(&basis, 128);
        let (mu_mp, bnorm_mp) = gram_schmidt_real::<rustmath_reals::RealMPFR>(&basis, 128);
        for i in 0..3 {
            assert!((bnorm_bf[i].to_f64() - exact_norm2(i)).abs() < 1e-12);
            assert!((bnorm_mp[i].to_f64() - exact_norm2(i)).abs() < 1e-12);
            for j in 0..i {
                let mq = mu_q[i][j].to_f64().unwrap();
                assert!((mu_bf[i][j].to_f64() - mq).abs() < 1e-12);
                assert!((mu_mp[i][j].to_f64() - mq).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rf_from_integer_is_exact_for_large_values() {
        // 2^100 + 12345 needs the multi-limb path; at 128 bits it is exactly
        // representable, so both backends must agree exactly with an
        // independent construction (exact BigFloat constructor / decimal
        // parsing for RealMPFR).
        let big = Integer::from(2).pow(100) + Integer::from(12345);
        let x: BigFloat = rf_from_integer(&big, 128);
        assert_eq!(x, BigFloat::from_integer(&big, 128));
        let y: rustmath_reals::RealMPFR = rf_from_integer(&big, 128);
        let y2 = <rustmath_reals::RealMPFR as RealField>::from_decimal_str(
            &big.to_string(),
            128,
        )
        .unwrap();
        assert_eq!(y, y2);
        // negative branch
        let z: BigFloat = rf_from_integer(&(-big.clone()), 128);
        assert_eq!(z, BigFloat::from_integer(&(-big), 128));
    }

    #[test]
    fn lattice_parent_and_gram() {
        let lat = Lattice::with_basis(vec![v(&[1, 0, 0]), v(&[0, 1, 0])]).unwrap();
        assert_eq!(lat.rank(), 2);
        assert_eq!(lat.ambient_dimension(), 3);
        assert_eq!(lat.name(), "Lattice of rank 2 in Z^3");

        // Gram matrix is the identity (orthonormal basis rows).
        let g = lat.gram_matrix();
        assert_eq!(*g.get(0, 0).unwrap(), Integer::from(1));
        assert_eq!(*g.get(0, 1).unwrap(), Integer::from(0));
        assert_eq!(lat.determinant_squared().unwrap(), Integer::from(1));

        // Membership: (2,-3,0) is in the lattice, (0,0,1) is not.
        assert!(lat.contains(&v(&[2, -3, 0])));
        assert!(!lat.contains(&v(&[0, 0, 1])));
        assert!(lat.generators().len() == 2);
        assert_eq!(lat.zero().unwrap(), v(&[0, 0, 0]));
    }

    #[test]
    fn short_vectors_of_Z2() {
        // Standard Z² lattice: vectors with norm² ≤ 1 are ±e1, ±e2 → 2 pairs.
        let lat = Lattice::with_basis(vec![v(&[1, 0]), v(&[0, 1])]).unwrap();
        let sv = lat.short_vectors(&Integer::from(1));
        assert_eq!(sv.len(), 2);
        for (_, nrm) in &sv {
            assert_eq!(*nrm, Integer::from(1));
        }

        // norm² ≤ 2 adds the four diagonal ±(1,±1) → 2 more pairs → 4 total.
        let sv2 = lat.short_vectors(&Integer::from(2));
        assert_eq!(sv2.len(), 4);
    }

    #[test]
    fn shortest_vectors_of_skew_basis() {
        // A skewed but rank-2 lattice equal to Z²; shortest are norm 1.
        let lat = Lattice::from_generators(vec![v(&[1, 1]), v(&[2, 1])]).unwrap();
        let sh = lat.shortest_vectors();
        assert!(!sh.is_empty());
        let minnrm = sh.iter().map(|(_, n)| n.clone()).min().unwrap();
        assert_eq!(minnrm, Integer::from(1));
    }
}
