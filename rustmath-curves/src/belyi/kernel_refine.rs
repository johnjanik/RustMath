//! E1a: mixed-precision kernel refinement with high-precision certificates.
//!
//! The `[2,12,5]` production solve needs the KERNEL of the dim-n preconditioned
//! modular-forms matrix `M = ρ^{n−r}(A−I)` (n up to 3001 at 400 bits) plus a
//! GAP certificate — not the full SVD. Full hp Jacobi ([`super::mp_svd`]) is
//! measured at ≥ 16 days there; this module replaces it with:
//!
//! 1. **Coarse stage** — a kernel-direction estimate at low precision: pure-Rust
//!    `f64` ([`CoarseStage::F64`]), pure-Rust double-double (~106-bit,
//!    [`CoarseStage::DoubleDouble`], the FP64-wall fallback), or externally
//!    supplied candidates ([`CoarseStage::External`], e.g. the optional
//!    `scripts/coarse_kernel_cupy.py` GPU helper — *never trusted*, see below).
//! 2. **Refinement** — preconditioned defect correction: each step is one
//!    coarse LU solve (the preconditioner) plus one FULL-precision residual
//!    matvec (rayon over rows). Non-kernel components contract by roughly the
//!    coarse unit roundoff per step (~15 digits/iteration for `f64`, ~31 for
//!    dd; measured ~13/iteration on the dim-500 planted case), down to a floor
//!    of `~σ_k(M)` itself — the residual of the true kernel — which no target
//!    below `σ_k` can beat (such targets end in an honest `Err`). The
//!    measured FAILURE MODE of a coarse stage (the FP64 wall) is
//!    **separation**, not rate: when several singular values sit below
//!    `ε_coarse·‖M‖`, the coarse factorization cannot tell the kernel from
//!    its sub-ε neighbors, candidates and corrections mix them, and the
//!    refinement stalls above target (see `fp64_wall_dd_fallback`). A stall
//!    triggers the f64 → dd escalation: fresh dd coarse candidates (coarse
//!    inverse iteration at ~106 bits separates anything above ~1e-30) plus a
//!    dd preconditioner. If the target still cannot be certified, the honest
//!    outcome is `Err` (a labeled resource refusal — never a weakened
//!    assertion); kernels needing separation below the dd floor would need a
//!    triple-double preconditioner (an extension point, deliberately not
//!    faked here).
//! 3. **Certificates** — computed from the refined vectors and the matrix
//!    ALONE, independent of how the vectors were found (so `f64`/dd/GPU
//!    candidates are *checked*, never trusted):
//!    * per-vector: a rigorous upper bound on `‖M v‖₂/‖v‖₂` (Higham-style
//!      γ-bounds on the correctly-rounded MPFR dot products; see
//!      [`certified_residual_upper`]),
//!    * subspace: `σ_k(M) ≤ ‖MV‖_F / σ_min(V)` (Courant–Fischer over
//!      `range(V)`), with `σ_min(V)² ≥ 1 − ‖V*V − I‖_F` certified the same way,
//!    * gap: a rigorous LOWER bound on `σ_{k+1}(M)` via a verified Cholesky of
//!      the deflated Gram matrix `M*M + c²VV* − τI` (Demmel/Higham backward
//!      error + Weyl for a rank-k PSD perturbation; see [`gap_certificate`]).
//!      The trial shift τ is LOCATED by cheap double-double Cholesky probes on
//!      a seed-independent geometric ladder from `‖H‖_F` down to the dd floor,
//!      then certified with O(1) full-precision attempts; the coarse σ̂_{k+1}
//!      Rayleigh estimate (or the caller's `gap_shift_seed`) is HEURISTIC and
//!      only seeds the fallback ladder for rungs below the dd floor; it is
//!      stored labeled as such.
//!
//! Numerics can certify NONZERO facts only (residual below a bound, σ above a
//! bound); nothing here ever claims an exact zero.
//!
//! **Determinism policy**: every parallel loop in this module partitions its
//! OUTPUT — each output scalar is produced by exactly one rayon task as a fixed
//! sequential fold of correctly-rounded MPFR (or IEEE) operations. No
//! fold/reduce reassociation is used anywhere, so all results (refinement
//! iterates, certificates, coarse LU) are bitwise reproducible across runs and
//! thread counts. This is asserted by tests, not just claimed.

use super::modular_forms_hp::ExtStream;
use super::mp_svd::{MpC, MpMatrix};
use num_complex::Complex64;
use rayon::prelude::*;
use rug::Float;

// ===========================================================================
// Error-free transforms (EFT) and double-double arithmetic
// ===========================================================================

/// Knuth two-sum: returns `(s, e)` with `s = RN(a+b)` and `s + e == a + b`
/// EXACTLY (no assumption on |a| vs |b|), for finite inputs with no overflow.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// Fast two-sum (Dekker): requires `|a| ≥ |b|` (or a = 0); same exactness.
#[inline]
pub fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// FMA two-prod: returns `(p, e)` with `p = RN(a·b)` and `p + e == a · b`
/// EXACTLY, provided `a·b` neither overflows nor has its error term fall below
/// the subnormal range (|a·b| ≳ 2^-969). `f64::mul_add` is a single correctly
/// rounded fused operation, so `e = fma(a, b, −p)` is the exact residue.
#[inline]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Double-double: an unevaluated sum `hi + lo` with `|lo| ≤ ulp(hi)/2`,
/// ~106 significant bits. Accurate (non-sloppy) add per the QD library.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

impl Dd {
    pub const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    pub const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };

    #[inline]
    pub fn from_f64(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }
    /// Round an MPFR value to dd: `hi = RN53(x)`, `lo = RN53(x − hi)`.
    pub fn from_float(x: &Float) -> Dd {
        let hi = x.to_f64();
        let lo = Float::with_val(x.prec(), x - hi).to_f64();
        Dd { hi, lo }
    }
    /// Exact lift into MPFR at `prec` (exact when `prec ≥ 107 + gap`; we use
    /// `hi + lo` as two exact additions at the target precision).
    pub fn to_float(self, prec: u32) -> Float {
        Float::with_val(prec, self.hi) + self.lo
    }
    #[inline]
    pub fn neg(self) -> Dd {
        Dd { hi: -self.hi, lo: -self.lo }
    }
    #[inline]
    pub fn add(self, b: Dd) -> Dd {
        let (s1, s2) = two_sum(self.hi, b.hi);
        let (t1, t2) = two_sum(self.lo, b.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (s1, s2) = quick_two_sum(s1, s2);
        Dd { hi: s1, lo: s2 }
    }
    #[inline]
    pub fn sub(self, b: Dd) -> Dd {
        self.add(b.neg())
    }
    #[inline]
    pub fn mul(self, b: Dd) -> Dd {
        let (p1, p2) = two_prod(self.hi, b.hi);
        let p2 = p2 + self.hi * b.lo + self.lo * b.hi;
        let (p1, p2) = quick_two_sum(p1, p2);
        Dd { hi: p1, lo: p2 }
    }
    /// Long division: three quotient terms then renormalize (~full dd accuracy).
    pub fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self.sub(b.mul(Dd::from_f64(q1)));
        let q2 = r.hi / b.hi;
        let r = r.sub(b.mul(Dd::from_f64(q2)));
        let q3 = r.hi / b.hi;
        let (s, e) = quick_two_sum(q1, q2);
        Dd { hi: s, lo: e }.add(Dd::from_f64(q3))
    }
    #[inline]
    pub fn abs(self) -> Dd {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) { self.neg() } else { self }
    }
    /// One Newton step on `x² = a` from the f64 sqrt (full dd accuracy).
    pub fn sqrt(self) -> Dd {
        if self.hi == 0.0 && self.lo == 0.0 {
            return Dd::ZERO;
        }
        let x = self.hi.sqrt();
        let xd = Dd::from_f64(x);
        // x + (a − x²)/(2x)
        let r = self.sub(xd.mul(xd));
        xd.add(Dd::from_f64(r.hi / (2.0 * x)))
    }
}

/// Complex double-double.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DdC {
    pub re: Dd,
    pub im: Dd,
}

// ===========================================================================
// Coarse scalar abstraction: the same LU / inverse-iteration code runs at
// f64 (Complex64) and double-double (DdC).
// ===========================================================================

/// Arithmetic surface the coarse stage needs, implemented for `Complex64`
/// (f64 coarse) and [`DdC`] (double-double coarse).
pub trait CoarseScalar: Copy + Send + Sync {
    fn zero() -> Self;
    fn from_f64(x: f64) -> Self;
    fn from_mpc(z: &MpC) -> Self;
    fn to_mpc(self, prec: u32) -> MpC;
    fn add(self, b: Self) -> Self;
    fn sub(self, b: Self) -> Self;
    fn mul(self, b: Self) -> Self;
    fn div(self, b: Self) -> Self;
    fn conj(self) -> Self;
    /// |z|² rounded to f64 — pivot comparisons and normalizations only.
    fn abs2_f64(self) -> f64;
    /// The unit roundoff of this arithmetic (2⁻⁵³ / 2⁻¹⁰⁵), for diagnostics.
    fn unit_roundoff() -> f64;
    const NAME: &'static str;
}

impl CoarseScalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn from_mpc(z: &MpC) -> Self {
        Complex64::new(z.re.to_f64(), z.im.to_f64())
    }
    fn to_mpc(self, prec: u32) -> MpC {
        MpC::from_f64(prec, self.re, self.im)
    }
    fn add(self, b: Self) -> Self {
        self + b
    }
    fn sub(self, b: Self) -> Self {
        self - b
    }
    fn mul(self, b: Self) -> Self {
        self * b
    }
    fn div(self, b: Self) -> Self {
        self / b
    }
    fn conj(self) -> Self {
        // NB: must be fully qualified — the by-value trait method would
        // otherwise shadow num_complex's &self inherent method (recursion).
        Complex64::new(self.re, -self.im)
    }
    fn abs2_f64(self) -> f64 {
        self.norm_sqr()
    }
    fn unit_roundoff() -> f64 {
        2f64.powi(-53)
    }
    const NAME: &'static str = "f64";
}

impl CoarseScalar for DdC {
    fn zero() -> Self {
        DdC { re: Dd::ZERO, im: Dd::ZERO }
    }
    fn from_f64(x: f64) -> Self {
        DdC { re: Dd::from_f64(x), im: Dd::ZERO }
    }
    fn from_mpc(z: &MpC) -> Self {
        DdC { re: Dd::from_float(&z.re), im: Dd::from_float(&z.im) }
    }
    fn to_mpc(self, prec: u32) -> MpC {
        MpC::new(self.re.to_float(prec), self.im.to_float(prec))
    }
    fn add(self, b: Self) -> Self {
        DdC { re: self.re.add(b.re), im: self.im.add(b.im) }
    }
    fn sub(self, b: Self) -> Self {
        DdC { re: self.re.sub(b.re), im: self.im.sub(b.im) }
    }
    fn mul(self, b: Self) -> Self {
        DdC {
            re: self.re.mul(b.re).sub(self.im.mul(b.im)),
            im: self.re.mul(b.im).add(self.im.mul(b.re)),
        }
    }
    fn div(self, b: Self) -> Self {
        // (a·conj b)/|b|²
        let den = b.re.mul(b.re).add(b.im.mul(b.im));
        let num = self.mul(b.conj());
        DdC { re: num.re.div(den), im: num.im.div(den) }
    }
    fn conj(self) -> Self {
        DdC { re: self.re, im: self.im.neg() }
    }
    fn abs2_f64(self) -> f64 {
        let a = self.re.mul(self.re).add(self.im.mul(self.im));
        a.hi + a.lo
    }
    fn unit_roundoff() -> f64 {
        2f64.powi(-105)
    }
    const NAME: &'static str = "dd";
}

// ===========================================================================
// Coarse LU (partial pivoting) + solves, rayon-threaded row updates
// ===========================================================================

/// Packed LU factorization `P A = L U` at coarse precision.
pub struct CoarseLu<T> {
    n: usize,
    a: Vec<T>,
    piv: Vec<usize>,
    /// True if a pivot was numerically zero and replaced by a tiny value —
    /// candidate quality may suffer; certificates are unaffected (they never
    /// consult the factorization).
    pub had_zero_pivot: bool,
}

fn lu_factor<T: CoarseScalar>(mut a: Vec<T>, n: usize) -> CoarseLu<T> {
    let mut piv = Vec::with_capacity(n);
    let mut had_zero_pivot = false;
    let mut max_abs2 = 0f64;
    for x in &a {
        max_abs2 = max_abs2.max(x.abs2_f64());
    }
    for j in 0..n {
        // pivot search in column j
        let mut best = j;
        let mut best_val = a[j * n + j].abs2_f64();
        for i in (j + 1)..n {
            let v = a[i * n + j].abs2_f64();
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        if best != j {
            for c in 0..n {
                a.swap(j * n + c, best * n + c);
            }
        }
        piv.push(best);
        if best_val == 0.0 {
            // exactly singular at this precision: substitute a tiny pivot so the
            // factorization completes (inverse iteration WANTS near-singularity).
            had_zero_pivot = true;
            let tiny = T::from_f64(f64::EPSILON * (max_abs2.sqrt() + 1.0));
            a[j * n + j] = tiny;
        }
        let pivot = a[j * n + j];
        // trailing update, rayon over rows (each row updated by one task,
        // sequentially left-to-right: bitwise deterministic).
        let (done, rest) = a.split_at_mut((j + 1) * n);
        let pivot_row = &done[j * n..(j + 1) * n];
        rest.par_chunks_mut(n).for_each(|row| {
            let l = row[j].div(pivot);
            row[j] = l;
            for c in (j + 1)..n {
                row[c] = row[c].sub(l.mul(pivot_row[c]));
            }
        });
    }
    CoarseLu { n, a, piv, had_zero_pivot }
}

impl<T: CoarseScalar> CoarseLu<T> {
    /// Solve `A x = b` in place (`x = U⁻¹ L⁻¹ P b`).
    fn solve(&self, b: &mut [T]) {
        let n = self.n;
        for j in 0..n {
            b.swap(j, self.piv[j]);
        }
        for i in 1..n {
            let mut acc = b[i];
            for j in 0..i {
                acc = acc.sub(self.a[i * n + j].mul(b[j]));
            }
            b[i] = acc;
        }
        for i in (0..n).rev() {
            let mut acc = b[i];
            for j in (i + 1)..n {
                acc = acc.sub(self.a[i * n + j].mul(b[j]));
            }
            b[i] = acc.div(self.a[i * n + i]);
        }
    }

    /// Solve `A* x = b` in place (`A* = U* L* P`, so `x = Pᵀ L⁻* U⁻* b`).
    fn solve_conj_transpose(&self, b: &mut [T]) {
        let n = self.n;
        // U* y = b (U* is lower triangular with conjugated entries)
        for i in 0..n {
            let mut acc = b[i];
            for j in 0..i {
                acc = acc.sub(self.a[j * n + i].conj().mul(b[j]));
            }
            b[i] = acc.div(self.a[i * n + i].conj());
        }
        // L* w = y (L* is unit upper triangular conjugated)
        for i in (0..n).rev() {
            let mut acc = b[i];
            for j in (i + 1)..n {
                acc = acc.sub(self.a[j * n + i].conj().mul(b[j]));
            }
            b[i] = acc;
        }
        // x = Pᵀ w: undo the row swaps in reverse order
        for j in (0..n).rev() {
            b.swap(j, self.piv[j]);
        }
    }
}

// ===========================================================================
// Row access abstraction: in-memory MpMatrix or the streamed EXT dump
// ===========================================================================

/// Row access at a chosen precision — the only interface the refinement,
/// certificates, and gap machinery use, so the streamed EXT path and the
/// in-memory path share every line of the math.
pub trait RowSource: Sync {
    fn dim(&self) -> usize;
    /// Row `i` with every entry at precision `prec`.
    fn row(&self, i: usize, prec: u32) -> Vec<MpC>;
}

impl RowSource for MpMatrix {
    fn dim(&self) -> usize {
        assert_eq!(self.rows, self.cols, "kernel refinement needs a square matrix");
        self.rows
    }
    fn row(&self, i: usize, prec: u32) -> Vec<MpC> {
        (0..self.cols)
            .map(|j| {
                let z = self.get(i, j);
                MpC::new(Float::with_val(prec, &z.re), Float::with_val(prec, &z.im))
            })
            .collect()
    }
}

impl RowSource for ExtStream {
    fn dim(&self) -> usize {
        self.dim
    }
    fn row(&self, i: usize, prec: u32) -> Vec<MpC> {
        ExtStream::row(self, i, prec).expect("EXT dump read failed mid-refinement")
    }
}

/// Materialize the coarse copy of the matrix (row-major) — O(n²) coarse
/// scalars (f64: 16 B/entry, dd: 32 B/entry; 3001² ≈ 144 / 288 MB).
fn coarse_copy<T: CoarseScalar, S: RowSource + ?Sized>(src: &S, prec: u32) -> Vec<T> {
    let n = src.dim();
    let mut out = vec![T::zero(); n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, chunk)| {
        let row = src.row(i, prec);
        for (c, z) in chunk.iter_mut().zip(row.iter()) {
            *c = T::from_mpc(z);
        }
    });
    out
}

// ===========================================================================
// hp vector helpers (all row/output-partitioned parallel ⇒ deterministic)
// ===========================================================================

fn dot_conj_hp(a: &[MpC], b: &[MpC], prec: u32) -> MpC {
    let mut acc = MpC::zero(prec);
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc.add(&x.conj_mul(y));
    }
    acc
}

fn norm2_hp(v: &[MpC], prec: u32) -> Float {
    let mut acc = Float::with_val(prec, 0);
    for x in v {
        acc += x.abs2();
    }
    acc.sqrt()
}

/// `M v` plus, per row, the magnitude sum `t̂_i = Σ_j (|re|+|im|)(|re|+|im|)`
/// which upper-bounds `Σ_j |m_ij||v_j|` — the certificate raw material.
/// Rayon over rows; each row is one task's sequential fold (deterministic).
fn matvec_with_abs<S: RowSource + ?Sized>(src: &S, v: &[MpC], prec: u32) -> (Vec<MpC>, Vec<Float>) {
    let n = src.dim();
    assert_eq!(v.len(), n);
    let pairs: Vec<(MpC, Float)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = src.row(i, prec);
            let mut acc = MpC::zero(prec);
            let mut t = Float::with_val(prec, 0);
            for j in 0..n {
                acc = acc.add(&row[j].mul(&v[j]));
                let rm = Float::with_val(prec, row[j].re.clone().abs() + row[j].im.clone().abs());
                let vm = Float::with_val(prec, v[j].re.clone().abs() + v[j].im.clone().abs());
                t += rm * vm;
            }
            (acc, t)
        })
        .collect();
    pairs.into_iter().unzip()
}

/// Plain threaded `M v` at `prec` (no certificate material): rayon over rows,
/// each output entry one task's sequential left-to-right fold — bitwise
/// deterministic, and bitwise identical to [`ExtStream::matvec`] on the same
/// entries (identical op sequence). Public: the determinism-policy reference
/// implementation.
pub fn matvec_hp<S: RowSource + ?Sized>(src: &S, v: &[MpC], prec: u32) -> Vec<MpC> {
    let n = src.dim();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let row = src.row(i, prec);
            let mut acc = MpC::zero(prec);
            for j in 0..n {
                acc = acc.add(&row[j].mul(&v[j]));
            }
            acc
        })
        .collect()
}

/// Modified Gram–Schmidt with one re-orthogonalization pass. If a vector
/// collapses (norm below 2^{−prec/2} after projection — a rank-deficient
/// candidate set), it is replaced by a deterministic LCG vector and the pass
/// restarts on it; the event is recorded in `notes`.
fn mgs_orthonormalize(vs: &mut [Vec<MpC>], prec: u32, notes: &mut Vec<String>) {
    let n = vs[0].len();
    let mut collapse_floor = Float::with_val(prec, 1);
    collapse_floor >>= prec / 2;
    let mut lcg: u64 = 0x9e3779b97f4a7c15;
    let next_f64 = |lcg: &mut u64| -> f64 {
        *lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*lcg >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    for j in 0..vs.len() {
        let mut attempts = 0;
        loop {
            for _pass in 0..2 {
                for t in 0..j {
                    let (head, tail) = vs.split_at_mut(j);
                    let proj = dot_conj_hp(&head[t], &tail[0], prec);
                    for i in 0..n {
                        let s = head[t][i].mul(&proj);
                        tail[0][i] = tail[0][i].sub(&s);
                    }
                }
            }
            let nrm = norm2_hp(&vs[j], prec);
            if nrm > collapse_floor {
                for i in 0..n {
                    vs[j][i] = vs[j][i].div_real(&nrm);
                }
                break;
            }
            attempts += 1;
            assert!(attempts < 8, "MGS could not complete an orthonormal set (n = {n})");
            notes.push(format!(
                "MGS: candidate {j} collapsed (norm ≤ 2^-{}), replaced by a deterministic LCG vector",
                prec / 2
            ));
            for i in 0..n {
                vs[j][i] = MpC::from_f64(prec, next_f64(&mut lcg), next_f64(&mut lcg));
            }
        }
    }
}

// ===========================================================================
// Certificates
// ===========================================================================

/// γ_m = m·u/(1−m·u) with u = 2^{1−prec} (a safe over-estimate of the MPFR
/// round-to-nearest unit roundoff 2^{−prec}). Panics if m·u ≥ 2⁻²⁰ — the
/// asymptotic regime every bound below assumes.
fn gamma(m: usize, prec: u32) -> Float {
    let mut u = Float::with_val(prec, 1);
    u >>= prec - 1;
    let mu = Float::with_val(prec, m as f64 * &u);
    assert!(
        mu.clone().to_f64() < 2f64.powi(-20),
        "certificate γ-bound out of its asymptotic regime: m·u = {mu} (m = {m}, prec = {prec})"
    );
    let denom = Float::with_val(prec, 1 - &mu);
    mu / denom
}

/// Rigorous upper bound on `‖M v‖₂ / ‖v‖₂` from a computed matvec.
///
/// Inputs: `s = fl(Mv)` and the per-row magnitude sums `t̂` from
/// [`matvec_with_abs`], plus `v` itself; everything at precision `prec`.
///
/// Bound derivation (Higham, *Accuracy and Stability of Numerical Algorithms*,
/// §3.1/§3.6, conservatively inflated — every inflation DIRECTION is toward a
/// larger bound):
/// * Each MPFR op is correctly rounded: relative error ≤ 2^{−prec} < u := 2^{1−prec}.
/// * A length-n complex dot product (our `MpC::mul` = 6 rounded ops/component,
///   plus n−1 complex adds) satisfies `|fl(s_i) − s_i| ≤ γ_{2n+4}·Σ_j |m_ij||v_j|`.
/// * `t̂_i` upper-bounds `Σ|m||v|` structurally ((|re|+|im|) ≥ |z| per factor)
///   but is itself rounded; the factor 2 covers its own `γ_{2n}` deflation many
///   times over. So `E_i := 2·γ_{2n+4}·t̂_i ≥ |fl(s_i) − (Mv)_i|`.
/// * `‖Mv‖ ≤ ‖fl(s)‖ + ‖E‖`, with the computed norms inflated by `(1+4γ_{n+4})`
///   (sum of squares + sqrt roundings) and the E-norm doubled.
/// * `‖v‖ ≥ fl(‖v‖)·(1 − 4γ_{n+4})`, and the final division adds one rounding,
///   covered by a last `(1+4u)` inflation.
///
/// At prec = 256, n ≤ 3001 the total inflation is ~1e-73 relative — the bound
/// is dominated by the true residual, as it must be.
pub fn certified_residual_upper(s: &[MpC], t_hat: &[Float], v: &[MpC], prec: u32) -> Float {
    let n = v.len();
    let g_dot = gamma(2 * n + 4, prec);
    let g_norm = gamma(n + 4, prec);
    let mut u = Float::with_val(prec, 1);
    u >>= prec - 1;

    let s_norm = norm2_hp(s, prec);
    let mut e2 = Float::with_val(prec, 0);
    for t in t_hat {
        let e = Float::with_val(prec, 2 * g_dot.clone() * t);
        e2 += e.clone() * e;
    }
    let e_norm = e2.sqrt();
    let infl = Float::with_val(prec, 1 + Float::with_val(prec, 4 * g_norm.clone()));
    let mv_up = Float::with_val(prec, s_norm * &infl) + Float::with_val(prec, 2 * e_norm);

    let v_norm = norm2_hp(v, prec);
    let defl = Float::with_val(prec, 1 - Float::with_val(prec, 4 * g_norm));
    let v_low = Float::with_val(prec, v_norm * defl);
    assert!(v_low > 0, "certified_residual_upper: ‖v‖ underflowed its deflation");

    let ratio = Float::with_val(prec, mv_up / v_low);
    ratio * Float::with_val(prec, 1 + Float::with_val(prec, 4 * u))
}

/// Rigorous upper bound on `‖V*V − I‖_F` for the column set `vs` (certificate
/// for `σ_min(V)² ≥ 1 − bound`); same γ machinery as
/// [`certified_residual_upper`].
fn certified_gram_defect(vs: &[Vec<MpC>], prec: u32) -> Float {
    let n = vs[0].len();
    let k = vs.len();
    let g_dot = gamma(2 * n + 4, prec);
    let g_norm = gamma(k * k + 4, prec);
    let mut f2 = Float::with_val(prec, 0);
    let mut e2 = Float::with_val(prec, 0);
    for (a, va) in vs.iter().enumerate() {
        for (b, vb) in vs.iter().enumerate() {
            let mut acc = MpC::zero(prec);
            let mut t = Float::with_val(prec, 0);
            for j in 0..n {
                acc = acc.add(&va[j].conj_mul(&vb[j]));
                let am = Float::with_val(prec, va[j].re.clone().abs() + va[j].im.clone().abs());
                let bm = Float::with_val(prec, vb[j].re.clone().abs() + vb[j].im.clone().abs());
                t += am * bm;
            }
            if a == b {
                acc = acc.sub(&MpC::one(prec));
            }
            f2 += acc.abs2();
            let e = Float::with_val(prec, 2 * g_dot.clone() * t);
            e2 += e.clone() * e;
        }
    }
    let f_norm = f2.sqrt();
    let e_norm = e2.sqrt();
    let infl = Float::with_val(prec, 1 + Float::with_val(prec, 4 * g_norm));
    Float::with_val(prec, f_norm * infl) + Float::with_val(prec, 2 * e_norm)
}

/// The rigorous gap certificate. Every field's epistemic status is explicit.
#[derive(Debug, Clone)]
pub struct GapCertificate {
    /// RIGOROUS: `σ_k(M) ≤ sigma_k_upper` (Courant–Fischer over `range(V)`:
    /// certified `‖MV‖_F` divided by certified `σ_min(V) ≥ √(1 − ‖V*V−I‖)`).
    pub sigma_k_upper: Float,
    /// RIGOROUS: `σ_{k+1}(M) ≥ sigma_next_lower`, from the verified Cholesky of
    /// `fl(M*M + c²VV*) − sI` (Demmel/Higham backward-error bound) and Weyl for
    /// the rank-k PSD deflation term. Zero if no shift could be verified.
    pub sigma_next_lower: Float,
    /// HEURISTIC ONLY: the shift-search seed — the coarse-stage Rayleigh
    /// estimate of σ_{k+1}, or `KernelRefineOptions::gap_shift_seed` when the
    /// caller supplied one. Used solely to seed the FALLBACK trial-shift
    /// ladder when the dd locator finds no rung (see [`gap_certificate`]).
    /// Never part of any certified claim.
    pub sigma_next_estimate_heuristic: f64,
    /// `sigma_next_lower > sigma_k_upper` — the certified separation verdict.
    pub separated: bool,
    /// The exact mathematical statement certified, with the bound constants.
    pub note: String,
}

/// Deflated Gram matrix `H = fl(M*M + c²VV*)` (upper triangle, stored by
/// column) at precision `p2`, plus a rigorous Frobenius bound on the formation
/// error `‖H − (M*M + c²VV*)‖_F ≤ bound_form`.
///
/// Per entry, `|ΔH_ij| ≤ γ_{2(n+k)+8}·(Σ_l |m_li||m_lj| + c²Σ_t |v_ti||v_tj|)`,
/// and by Cauchy–Schwarz the Frobenius total telescopes to
/// `γ·(‖M‖_F² + c²‖V‖_F²)`; the factor 2 covers the rounding of the norm
/// accumulations themselves.
///
/// Row-streamed: one sequential pass over the rows of `M`; within each row the
/// update is rayon-parallel over COLUMNS of `H` (disjoint output ⇒ bitwise
/// deterministic).
fn gram_deflated<S: RowSource + ?Sized>(
    src: &S,
    vs: &[Vec<MpC>],
    c: &Float,
    p2: u32,
) -> (Vec<Vec<MpC>>, Float) {
    let n = src.dim();
    let k = vs.len();
    let mut cols: Vec<Vec<MpC>> = (0..n).map(|j| vec![MpC::zero(p2); j + 1]).collect();
    let mut m_frob2 = Float::with_val(p2, 0);
    for l in 0..n {
        let row = src.row(l, p2);
        cols.par_iter_mut().enumerate().for_each(|(j, col)| {
            for i in 0..=j {
                let term = row[i].conj_mul(&row[j]);
                col[i] = col[i].add(&term);
            }
        });
        for z in &row {
            m_frob2 += z.abs2();
        }
    }
    let c2 = Float::with_val(p2, c.clone() * c);
    let mut v_frob2 = Float::with_val(p2, 0);
    for vt in vs {
        for z in vt.iter() {
            v_frob2 += Float::with_val(p2, z.abs2());
        }
    }
    cols.par_iter_mut().enumerate().for_each(|(j, col)| {
        for i in 0..=j {
            let mut acc = MpC::zero(p2);
            for vt in vs {
                let vi = MpC::new(Float::with_val(p2, &vt[i].re), Float::with_val(p2, &vt[i].im));
                let vj = MpC::new(Float::with_val(p2, &vt[j].re), Float::with_val(p2, &vt[j].im));
                acc = acc.add(&vj.conj_mul(&vi));
            }
            col[i] = col[i].add(&acc.scale(&c2));
        }
    });
    let g = gamma(2 * (n + k) + 8, p2);
    let bound_form =
        Float::with_val(p2, 2 * g) * (m_frob2 + Float::with_val(p2, c2 * v_frob2));
    (cols, bound_form)
}

/// Right-looking complex Cholesky attempt on the Hermitian matrix given by its
/// upper triangle (by column), with the diagonal shifted by `−s`. On success
/// returns `‖R‖_F²` (for the backward-error bound); on failure (a non-positive
/// reduced diagonal) returns `None`. Trailing updates are rayon over rows.
fn cholesky_shifted(cols: &[Vec<MpC>], s: &Float, p2: u32) -> Option<Float> {
    let n = cols.len();
    // Full working Hermitian matrix W (row-major), W = H − sI.
    let mut w = MpMatrix::zeros(n, n, p2);
    for (j, col) in cols.iter().enumerate() {
        for (i, z) in col.iter().enumerate() {
            w.set(i, j, z.clone());
            if i != j {
                w.set(j, i, z.conj());
            }
        }
        let d = Float::with_val(p2, &col[j].re - s);
        w.set(j, j, MpC::new(d, Float::with_val(p2, 0)));
    }
    let mut r_frob2 = Float::with_val(p2, 0);
    for jstep in 0..n {
        let d = w.get(jstep, jstep).re.clone();
        if d <= 0 {
            return None;
        }
        let rjj = d.clone().sqrt();
        // R row jstep: r_j = W[jstep, jstep+1..]/rjj, then trailing update.
        let mut rrow = vec![MpC::zero(p2); n];
        for c in (jstep + 1)..n {
            rrow[c] = w.get(jstep, c).div_real(&rjj);
        }
        r_frob2 += d;
        for z in rrow.iter().skip(jstep + 1) {
            r_frob2 += z.abs2();
        }
        let cols_n = w.cols;
        let (_, tail) = w.data.split_at_mut((jstep + 1) * cols_n);
        tail.par_chunks_mut(cols_n).enumerate().for_each(|(off, row)| {
            let i = jstep + 1 + off;
            let ri = rrow[i].clone();
            for c in i..n {
                let t = ri.conj_mul(&rrow[c]);
                row[c] = row[c].sub(&t);
            }
        });
        // (lower triangle of the trailing block is never read)
    }
    Some(r_frob2)
}

/// LOCATOR probe (heuristic, NO soundness burden): a double-double (~106-bit)
/// shifted Cholesky attempt on the dd rounding of the deflated Gram triangle.
/// Success is evidence — not proof — that `λ_min(H) ≳ s` (the dd rounding of
/// `H` alone perturbs eigenvalues by ~2⁻¹⁰⁵·‖H‖); the caller must certify any
/// located rung with the full-precision verified [`cholesky_shifted`] before
/// claiming anything. A wrong answer here can only waste one hp attempt or
/// miss a rung, never produce an unsound certificate.
///
/// Same right-looking algorithm and rayon row-partitioning as
/// [`cholesky_shifted`] (deterministic); non-finite reduced diagonals count as
/// failure (dd overflow on garbage input must not "succeed").
fn cholesky_shifted_dd_probe(cols: &[Vec<DdC>], s: f64) -> bool {
    let n = cols.len();
    let shift = Dd::from_f64(s);
    let zero = DdC { re: Dd::ZERO, im: Dd::ZERO };
    // Full working Hermitian matrix W (row-major), W = H_dd − sI.
    let mut w: Vec<DdC> = vec![zero; n * n];
    for (j, col) in cols.iter().enumerate() {
        for (i, z) in col.iter().enumerate() {
            w[i * n + j] = *z;
            if i != j {
                w[j * n + i] = z.conj();
            }
        }
        let d = col[j].re.sub(shift);
        w[j * n + j] = DdC { re: d, im: Dd::ZERO };
    }
    for jstep in 0..n {
        let d = w[jstep * n + jstep].re;
        let dv = d.hi + d.lo;
        if !(dv > 0.0) || !dv.is_finite() {
            return false;
        }
        let rjj = d.sqrt();
        let mut rrow: Vec<DdC> = vec![zero; n];
        for c in (jstep + 1)..n {
            let z = w[jstep * n + c];
            rrow[c] = DdC { re: z.re.div(rjj), im: z.im.div(rjj) };
        }
        let (_, tail) = w.split_at_mut((jstep + 1) * n);
        tail.par_chunks_mut(n).enumerate().for_each(|(off, row)| {
            let i = jstep + 1 + off;
            let ri = rrow[i];
            for c in i..n {
                row[c] = row[c].sub(ri.conj().mul(rrow[c]));
            }
        });
    }
    true
}

/// Build the full gap certificate for the refined set `vs` against `src`.
///
/// Certified statements (see [`GapCertificate`] field docs):
/// * `σ_k(M) ≤ ‖MV‖_F^up / √(1 − d_up)` where `d_up ≥ ‖V*V − I‖_F` — valid by
///   Courant–Fischer applied to the k-dimensional subspace `range(V)` whenever
///   `d_up < 1` (then `σ_min(V)² ≥ 1 − d_up > 0`).
/// * `σ_{k+1}(M) ≥ √(s − β_chol − β_form)` when the floating Cholesky of
///   `fl(H) − sI`, `H = M*M + c²VV*`, runs to completion at precision `p2`:
///   completion implies `fl(H) − sI + ΔC ⪰ 0` with
///   `‖ΔC‖₂ ≤ β_chol := 2γ_{4(n+1)}·‖R‖_F²` (Higham ASNA Thm 10.3, complex
///   constants absorbed by the 4× inflation), the formation error gives
///   `H ⪰ fl(H) − β_form·I`, hence `H ⪰ (s − β_chol − β_form)I`; and Weyl for
///   the rank-k PSD term `c²VV*` gives `μ_{k+1}(M*M) ≥ μ_min(H)`, i.e.
///   `σ_{k+1}(M)² ≥ s − β_chol − β_form`. This holds for ANY V — a bad V only
///   makes the Cholesky fail, never an unsound certificate.
///
/// Shift search (two stages, both HEURISTIC steering around the sound
/// verified-Cholesky core above — a bad search can only MISS a certificate,
/// never fake one):
/// 1. **dd LOCATOR**: cheap double-double Cholesky probes
///    ([`cholesky_shifted_dd_probe`]) walk a long geometric ladder of trial
///    shifts from `‖H‖_F` downward by factors of 4 (≥ `λ_min(H)` for PSD `H`,
///    so the ladder top is always above the rung) until a probe succeeds or
///    the ladder hits the dd resolution floor `‖H‖_F·2⁻¹⁰⁰` (~50 rungs ≈ 30
///    decades — tied to the locator's ~106-bit precision: dd cannot resolve
///    `λ_min` below ~2⁻¹⁰⁵·‖H‖). The first accepting rung is then CERTIFIED
///    by at most two full-precision attempts (the rung and one safety
///    quartering, covering dd-optimistic boundary calls) — O(1) hp Cholesky
///    attempts regardless of where the rung lies, independent of any seed.
/// 2. **Seed-ladder fallback**: only if the locator finds no rung (σ_{k+1}²
///    below the dd floor, e.g. the FP64-wall shapes), the trial shift starts
///    at `(seed/2)²` and quarters on failure, up to 8 hp attempts, where
///    `sigma_next_estimate` is the caller's seed — the coarse HEURISTIC
///    σ̂_{k+1}, or `KernelRefineOptions::gap_shift_seed` when supplied.
///
/// Only after BOTH stages fail is the labeled no-certificate outcome
/// (`sigma_next_lower = 0`) returned.
pub fn gap_certificate<S: RowSource + ?Sized>(
    src: &S,
    vs: &[Vec<MpC>],
    prec: u32,
    p2: u32,
    sigma_next_estimate: f64,
) -> GapCertificate {
    let n = src.dim();
    let k = vs.len();

    // --- rigorous σ_k upper bound -----------------------------------------
    let mut mv_f2 = Float::with_val(prec, 0);
    for v in vs {
        let (s, t) = matvec_with_abs(src, v, prec);
        let up = certified_residual_upper(&s, &t, v, prec);
        // columns are ~unit; ‖Mv_j‖ ≤ up·‖v_j‖ ≤ up·(1+d) — fold the (1+d)
        // slack into the final σ_min(V) division below by using ‖MV‖_F ≤
        // √(Σ up_j²)·max_j‖v_j‖ and bounding max‖v_j‖² ≤ 1 + d_up.
        mv_f2 += up.clone() * up;
    }
    let d_up = certified_gram_defect(vs, prec);
    let one = Float::with_val(prec, 1);
    let (sigma_k_upper, sk_note) = if d_up < 1 {
        let smin2 = Float::with_val(prec, &one - &d_up);
        let vmax2 = Float::with_val(prec, &one + &d_up);
        // ‖MV‖_F ≤ √(Σ up_j²·‖v_j‖²) ≤ √(Σ up_j²)·√(1+d_up)
        let num = Float::with_val(prec, mv_f2 * vmax2).sqrt();
        let den = smin2.sqrt();
        let mut u = Float::with_val(prec, 1);
        u >>= prec - 1;
        // Rounding count on this combination path (everything AFTER the
        // already-certified up_j and d_up; each MPFR op is correctly rounded
        // with relative error < u = 2^{1-prec}, itself a 2× over-estimate of
        // the true roundoff 2^{-prec}):
        //   mv_f2 accumulation:  k mults (up_j·up_j) + k adds  = 2k
        //   1 − d_up and 1 + d_up:                               2
        //   num: the mv_f2·vmax2 mult + its sqrt:                2
        //   den: smin2.sqrt():                                   1
        //   the num/den division:                                1
        //   the inflation multiply itself (below):               1
        // total m = 2k + 7 roundings, so the computed ratio can undershoot the
        // exact value by at most (1+u)^m. Inflate by (1 + 2m·u) ≥ (1+u)^m
        // (valid for m·u ≤ 1/2, which holds by orders of magnitude at any
        // usable prec) — a strict cover with a further 2× margin on top of the
        // 2× in u. The previous constant (1+4u) covered only 4 of these
        // roundings — a deficit, though ~1e-74 relative at prec = 256.
        let m_rounds = 2 * k + 7;
        let infl =
            Float::with_val(prec, 1 + Float::with_val(prec, (2 * m_rounds) as f64 * &u));
        (Float::with_val(prec, num / den) * infl, String::new())
    } else {
        (
            Float::with_val(prec, f64::INFINITY),
            format!(" σ_k bound VACUOUS: ‖V*V−I‖ bound {d_up} ≥ 1 (V not near-orthonormal)."),
        )
    };

    // --- rigorous σ_{k+1} lower bound -------------------------------------
    let c = {
        // any c is sound; c ~ 2‖M‖_F lifts the deflated directions well clear.
        let mut acc = Float::with_val(p2, 0);
        for i in 0..n {
            let row = src.row(i, p2);
            for z in &row {
                acc += z.abs2();
            }
        }
        Float::with_val(p2, 2 * acc.sqrt())
    };
    let (cols, bound_form) = gram_deflated(src, vs, &c, p2);
    let g_chol = gamma(4 * (n + 1), p2);
    let mut sigma_next_lower = Float::with_val(prec, 0);

    // Stage 1 — dd LOCATOR (heuristic, cheap): find the shift rung with
    // double-double probes before spending ANY full-precision Cholesky. The
    // coarse σ̂_{k+1} seed can be arbitrarily wrong (the coarse MGS destroys
    // the (k+1)-th direction whenever all deflated σ sit below the coarse
    // epsilon), so the locator ladder must not depend on it: it starts at
    // ‖H‖_F ≥ λ_min(H) and quarters down to the dd resolution floor.
    let mut h_frob2 = Float::with_val(p2, 0);
    for (j, col) in cols.iter().enumerate() {
        for (i, z) in col.iter().enumerate() {
            let a2 = z.abs2();
            h_frob2 += &a2;
            if i != j {
                // the stored triangle omits the mirrored lower entry
                h_frob2 += a2;
            }
        }
    }
    let h_frob = h_frob2.sqrt().to_f64();
    let mut located: Option<f64> = None;
    let mut dd_rungs = 0usize;
    if h_frob.is_finite() && h_frob > 0.0 {
        // Floor tied to the locator's precision: dd resolves λ_min only down
        // to ~2⁻¹⁰⁵·‖H‖, so rungs below ‖H‖_F·2⁻¹⁰⁰ are dd-invisible; those
        // are left to the seed fallback (stage 2b) or, failing that, the
        // labeled no-certificate outcome.
        let cols_dd: Vec<Vec<DdC>> =
            cols.iter().map(|col| col.iter().map(DdC::from_mpc).collect()).collect();
        let floor = h_frob * 2f64.powi(-100);
        let mut s = h_frob;
        while s > floor {
            dd_rungs += 1;
            if cholesky_shifted_dd_probe(&cols_dd, s) {
                located = Some(s);
                break;
            }
            s /= 4.0;
        }
    }

    // Stage 2 — full-precision CERTIFICATION (the only stage anything is
    // believed from). (a) located rung: the rung itself plus one safety
    // quartering (dd rounding of H can make the boundary call optimistic by
    // ~2⁻¹⁰⁵·‖H‖) — O(1) hp attempts. (b) no rung located: the seed ladder
    // from (seed/2)², quartering up to 8 attempts (the pre-locator behavior)
    // — the seed (coarse heuristic or the caller's gap_shift_seed) is the
    // only remaining information for rungs below the dd floor.
    let est_seed = if sigma_next_estimate.is_finite() && sigma_next_estimate > 0.0 {
        sigma_next_estimate
    } else {
        1.0
    };
    let mut hp_shifts: Vec<Float> = Vec::new();
    if let Some(s_loc) = located {
        hp_shifts.push(Float::with_val(p2, s_loc));
        hp_shifts.push(Float::with_val(p2, s_loc / 4.0));
    } else {
        let seed_shift = Float::with_val(p2, est_seed / 2.0);
        let mut shift = Float::with_val(p2, seed_shift.clone() * &seed_shift);
        for _ in 0..8 {
            if shift.clone().to_f64() <= 0.0 {
                break;
            }
            hp_shifts.push(shift.clone());
            shift = Float::with_val(p2, &shift / 4);
        }
    }
    let locator_note = match located {
        Some(s_loc) => format!(
            " dd locator: rung s ≈ {s_loc:.6e} located after {dd_rungs} dd probe(s) from ‖H‖_F ≈ {h_frob:.3e}."
        ),
        None => format!(
            " dd locator: NO rung above the dd floor ‖H‖_F·2^-100 ({dd_rungs} dd probe(s) from ‖H‖_F ≈ {h_frob:.3e}); fell back to the seed ladder."
        ),
    };
    let mut chol_note = String::from(" Cholesky: no shift verified.");
    for shift in &hp_shifts {
        if let Some(r_frob2) = cholesky_shifted(&cols, shift, p2) {
            let beta_chol = Float::with_val(p2, 2 * g_chol.clone()) * r_frob2;
            let mu = Float::with_val(p2, shift - &beta_chol) - &bound_form;
            if mu > 0 {
                sigma_next_lower = Float::with_val(prec, mu.sqrt_ref());
                chol_note = format!(
                    " Cholesky verified at shift s = {:.6e} (β_chol + β_form = {:.3e}).",
                    shift.to_f64(),
                    (Float::with_val(p2, &beta_chol + &bound_form)).to_f64()
                );
            }
            // success with μ ≤ 0: smaller shifts only shrink μ further — stop.
            break;
        }
    }

    let separated = sigma_next_lower > sigma_k_upper;
    let note = format!(
        "RIGOROUS: σ_{k}(M) ≤ {:.6e} via Courant–Fischer on range(V) with certified \
         ‖MV‖_F (γ_(2n+4) dot bounds, u = 2^(1-{prec})) and σ_min(V)² ≥ 1 − {:.3e}.{sk_note} \
         RIGOROUS: σ_{}(M) ≥ {:.6e} via verified complex Cholesky of fl(M*M + c²VV*) − sI \
         at {p2} bits (backward error ≤ 2γ_(4(n+1))·‖R‖_F², formation ≤ {:.3e}; \
         Weyl rank-{k} PSD deflation).{locator_note}{chol_note} \
         HEURISTIC (labeled, not certified): the dd locator ladder and the seed σ̂_{} ≈ {:.3e} \
         only steer WHICH s is tried. n = {n}.",
        sigma_k_upper.to_f64(),
        d_up.to_f64(),
        k + 1,
        sigma_next_lower.to_f64(),
        bound_form.to_f64(),
        k + 1,
        sigma_next_estimate,
    );
    GapCertificate {
        sigma_k_upper,
        sigma_next_lower,
        sigma_next_estimate_heuristic: sigma_next_estimate,
        separated,
        note,
    }
}

// ===========================================================================
// Coarse stage: block inverse iteration on M*M via LU of M
// ===========================================================================

struct CoarseResult {
    /// k candidate right-singular directions, lifted to hp.
    candidates: Vec<Vec<MpC>>,
    /// Heuristic Rayleigh estimates σ̂_1 ≤ … ≤ σ̂_{k+g} (coarse arithmetic).
    sigma_estimates: Vec<f64>,
    note: String,
}

fn coarse_stage<T: CoarseScalar, S: RowSource + ?Sized>(
    src: &S,
    k: usize,
    extra: usize,
    iters: usize,
    prec: u32,
) -> (CoarseResult, CoarseLu<T>) {
    let n = src.dim();
    let kg = (k + extra).min(n);
    let m: Vec<T> = coarse_copy(src, prec);
    let lu = lu_factor(m.clone(), n);

    // deterministic LCG start block
    let mut lcg: u64 = 0x243f6a8885a308d3;
    let next = |lcg: &mut u64| -> f64 {
        *lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*lcg >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    let mut xs: Vec<Vec<T>> = (0..kg)
        .map(|_| {
            (0..n)
                .map(|_| {
                    let re = next(&mut lcg);
                    let im = next(&mut lcg);
                    T::from_mpc(&MpC::from_f64(53, re, im))
                })
                .collect()
        })
        .collect();

    // block inverse iteration on (M*M)^{-1}: x ← M⁻¹ M⁻* x, MGS between steps
    let mgs_coarse = |xs: &mut Vec<Vec<T>>| {
        for j in 0..xs.len() {
            for t in 0..j {
                let mut proj = T::zero();
                for i in 0..n {
                    proj = proj.add(xs[t][i].conj().mul(xs[j][i]));
                }
                for i in 0..n {
                    let s = xs[t][i].mul(proj);
                    xs[j][i] = xs[j][i].sub(s);
                }
            }
            let mut nrm2 = 0f64;
            for i in 0..n {
                nrm2 += xs[j][i].abs2_f64();
            }
            let scale = T::from_f64(1.0 / nrm2.sqrt().max(f64::MIN_POSITIVE));
            for i in 0..n {
                xs[j][i] = xs[j][i].mul(scale);
            }
        }
    };
    for _ in 0..iters {
        for x in xs.iter_mut() {
            lu.solve_conj_transpose(x);
            lu.solve(x);
        }
        mgs_coarse(&mut xs);
    }

    // Rayleigh estimates σ̂_j = ‖M x_j‖ (coarse matvec), then sort ascending.
    let sigma_of = |x: &Vec<T>| -> f64 {
        let mut nrm2 = 0f64;
        for i in 0..n {
            let mut acc = T::zero();
            for j in 0..n {
                acc = acc.add(m[i * n + j].mul(x[j]));
            }
            nrm2 += acc.abs2_f64();
        }
        nrm2.sqrt()
    };
    let mut order: Vec<(f64, usize)> = xs.iter().map(sigma_of).zip(0..kg).collect();
    order.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let sigma_estimates: Vec<f64> = order.iter().map(|p| p.0).collect();
    let candidates: Vec<Vec<MpC>> = order[..k]
        .iter()
        .map(|&(_, idx)| xs[idx].iter().map(|z| z.to_mpc(prec)).collect())
        .collect();
    let note = format!(
        "coarse stage: {} block inverse iteration ({} iters, {} vectors{}), σ̂ = {:?}",
        T::NAME,
        iters,
        kg,
        if lu.had_zero_pivot { ", zero pivot patched" } else { "" },
        sigma_estimates.iter().map(|s| format!("{s:.3e}")).collect::<Vec<_>>(),
    );
    (CoarseResult { candidates, sigma_estimates, note }, lu)
}

// ===========================================================================
// The refinement driver
// ===========================================================================

/// Which coarse stage supplies candidates (and the default preconditioner).
pub enum CoarseStage {
    /// Pure-Rust f64 (sufficient while the small σ sit above the f64 floor).
    F64,
    /// Pure-Rust double-double (~106-bit) — the FP64-wall fallback.
    DoubleDouble,
    /// Externally supplied candidates (e.g. the cupy GPU helper), as decimal
    /// (re, im) strings per component. UNTRUSTED input: used only as starting
    /// points; every certificate is recomputed from scratch in hp.
    External { candidates: Vec<Vec<(String, String)>> },
}

/// Options for [`refine_kernel`] / [`refine_kernel_streamed`].
pub struct KernelRefineOptions {
    /// Target/working precision in bits for refinement and certificates.
    pub prec_bits: u32,
    /// The certificate target: refinement succeeds when every per-vector
    /// certified `‖Mv‖/‖v‖` upper bound is ≤ this decimal value.
    pub target_residual_decimal: String,
    /// Iteration budget; exhausting it is an HONEST `Err` (resource refusal),
    /// never a silently weakened result.
    pub max_iters: usize,
    /// Coarse candidate source.
    pub coarse: CoarseStage,
    /// Inverse-iteration steps inside the coarse stage.
    pub coarse_iters: usize,
    /// Extra coarse vectors beyond k (for the heuristic σ̂_{k+1} estimate).
    pub extra_coarse_vectors: usize,
    /// Compute the gap certificate after convergence.
    pub with_gap: bool,
    /// Precision for the gap Gram/Cholesky stage (default: `prec_bits`).
    pub gap_prec_bits: Option<u32>,
    /// Allow the automatic f64 → dd preconditioner escalation on stall (the
    /// FP64-wall fallback). Disable to observe/test the f64-only behavior.
    pub allow_dd_escalation: bool,
    /// Optional seed (in σ units) for the gap certificate's shift search,
    /// REPLACING the coarse-stage heuristic σ̂_{k+1} estimate. The seed matters
    /// when the dd shift LOCATOR (see [`gap_certificate`]) cannot see the rung
    /// — σ_{k+1}² below the dd resolution floor — and the fallback seed ladder
    /// is the only remaining information; a destroyed coarse heuristic there
    /// yields the labeled no-certificate outcome unless a caller who knows
    /// σ_{k+1} supplies it here. Heuristic steering ONLY: a wrong seed can
    /// cause a missing certificate, never an unsound one (every bound passes
    /// through the verified hp Cholesky).
    pub gap_shift_seed: Option<f64>,
}

impl KernelRefineOptions {
    pub fn new(prec_bits: u32, target_residual_decimal: &str) -> Self {
        KernelRefineOptions {
            prec_bits,
            target_residual_decimal: target_residual_decimal.to_string(),
            max_iters: 40,
            coarse: CoarseStage::F64,
            coarse_iters: 3,
            extra_coarse_vectors: 1,
            with_gap: true,
            gap_prec_bits: None,
            allow_dd_escalation: true,
            gap_shift_seed: None,
        }
    }
}

/// A refined kernel with its certificates. `vectors` are orthonormal at
/// `prec_bits`; `residual_bounds[j]` is a RIGOROUS upper bound on
/// `‖M v_j‖₂/‖v_j‖₂` (see [`certified_residual_upper`]); `gap` is the
/// rigorous/heuristic-labeled gap certificate. `provenance` records how the
/// vectors were FOUND — it is diagnostic only and carries no certified claim.
pub struct CertifiedKernel {
    pub vectors: Vec<Vec<MpC>>,
    pub residual_bounds: Vec<Float>,
    pub gap: Option<GapCertificate>,
    pub iterations: usize,
    pub provenance: String,
}

enum Precond {
    F64(CoarseLu<Complex64>),
    Dd(CoarseLu<DdC>),
}

impl Precond {
    /// Solve `M_c d = r/‖r‖` at coarse precision and return `‖r‖·d` lifted to hp.
    fn correction(&self, r: &[MpC], prec: u32) -> Vec<MpC> {
        let rn = norm2_hp(r, prec);
        if rn == 0 {
            return vec![MpC::zero(prec); r.len()];
        }
        let scaled: Vec<MpC> = r.iter().map(|z| z.div_real(&rn)).collect();
        match self {
            Precond::F64(lu) => {
                let mut b: Vec<Complex64> = scaled.iter().map(Complex64::from_mpc).collect();
                lu.solve(&mut b);
                b.into_iter().map(|z| z.to_mpc(prec).scale(&rn)).collect()
            }
            Precond::Dd(lu) => {
                let mut b: Vec<DdC> = scaled.iter().map(DdC::from_mpc).collect();
                lu.solve(&mut b);
                b.into_iter().map(|z| z.to_mpc(prec).scale(&rn)).collect()
            }
        }
    }
}

fn refine_core<S: RowSource + ?Sized>(
    src: &S,
    k: usize,
    opt: &KernelRefineOptions,
) -> Result<CertifiedKernel, String> {
    let n = src.dim();
    let prec = opt.prec_bits;
    if k == 0 || k >= n {
        return Err(format!("refine_kernel: k = {k} out of range for dim {n}"));
    }
    let target = Float::with_val(
        prec,
        Float::parse(&opt.target_residual_decimal)
            .map_err(|e| format!("bad target_residual_decimal: {e:?}"))?,
    );
    if target <= 0 {
        return Err("target residual must be positive (numerics cannot certify an exact zero)".into());
    }

    let mut notes: Vec<String> = Vec::new();
    // --- coarse stage ------------------------------------------------------
    let (mut vs, mut sigma_next_est, mut precond): (Vec<Vec<MpC>>, f64, Precond) = match &opt.coarse {
        CoarseStage::F64 => {
            let (cr, lu) = coarse_stage::<Complex64, S>(src, k, opt.extra_coarse_vectors, opt.coarse_iters, prec);
            notes.push(cr.note.clone());
            let est = cr.sigma_estimates.get(k).copied().unwrap_or(f64::NAN);
            (cr.candidates, est, Precond::F64(lu))
        }
        CoarseStage::DoubleDouble => {
            let (cr, lu) = coarse_stage::<DdC, S>(src, k, opt.extra_coarse_vectors, opt.coarse_iters, prec);
            notes.push(cr.note.clone());
            let est = cr.sigma_estimates.get(k).copied().unwrap_or(f64::NAN);
            (cr.candidates, est, Precond::Dd(lu))
        }
        CoarseStage::External { candidates } => {
            if candidates.len() != k {
                return Err(format!(
                    "external coarse stage supplied {} candidates, need {k}",
                    candidates.len()
                ));
            }
            let mut vs = Vec::with_capacity(k);
            for cand in candidates {
                if cand.len() != n {
                    return Err(format!(
                        "external candidate has {} components, need {n}",
                        cand.len()
                    ));
                }
                let mut v = Vec::with_capacity(n);
                for (re, im) in cand {
                    let fre = Float::parse(re).map_err(|e| format!("bad candidate re: {e:?}"))?;
                    let fim = Float::parse(im).map_err(|e| format!("bad candidate im: {e:?}"))?;
                    v.push(MpC::new(Float::with_val(prec, fre), Float::with_val(prec, fim)));
                }
                vs.push(v);
            }
            notes.push(format!("coarse stage: EXTERNAL ({k} untrusted candidates; certificates recomputed in hp)"));
            // still need σ̂_{k+1} for the gap shift + a preconditioner: run the
            // f64 stage for both (its candidates are discarded).
            let (cr, lu) = coarse_stage::<Complex64, S>(src, k, opt.extra_coarse_vectors, opt.coarse_iters, prec);
            let est = cr.sigma_estimates.get(k).copied().unwrap_or(f64::NAN);
            (vs, est, Precond::F64(lu))
        }
    };
    mgs_orthonormalize(&mut vs, prec, &mut notes);

    // --- refinement loop ---------------------------------------------------
    let mut prev_worst: Option<Float> = None;
    let mut stall_count = 0usize;
    let mut escalated = false;
    let mut iterations;
    for iter in 0..opt.max_iters {
        iterations = iter + 1;
        // full-precision residual matvecs + certificates
        let mut residuals: Vec<Vec<MpC>> = Vec::with_capacity(k);
        let mut bounds: Vec<Float> = Vec::with_capacity(k);
        for v in &vs {
            let (s, t) = matvec_with_abs(src, v, prec);
            bounds.push(certified_residual_upper(&s, &t, v, prec));
            residuals.push(s);
        }
        let worst = bounds.iter().cloned().fold(Float::with_val(prec, 0), |a, b| if b > a { b } else { a });
        if bounds.iter().all(|b| *b <= target) {
            let gap = if opt.with_gap {
                let p2 = opt.gap_prec_bits.unwrap_or(prec);
                // A user-supplied gap_shift_seed REPLACES the coarse heuristic
                // as the shift-search seed (it is recorded in the certificate's
                // heuristic slot — both are heuristic steering, never certified).
                let seed = match opt.gap_shift_seed {
                    Some(s) => {
                        notes.push(format!(
                            "gap shift seed OVERRIDDEN by caller: {s:.3e} (coarse heuristic was {sigma_next_est:.3e})"
                        ));
                        s
                    }
                    None => sigma_next_est,
                };
                Some(gap_certificate(src, &vs, prec, p2, seed))
            } else {
                None
            };
            notes.push(format!("converged after {iterations} iteration(s), worst certified residual {:.3e}", worst.to_f64()));
            return Ok(CertifiedKernel {
                vectors: vs,
                residual_bounds: bounds,
                gap,
                iterations,
                provenance: notes.join("; "),
            });
        }
        // stall detection: no half-decade of progress on the worst bound
        if let Some(prev) = &prev_worst {
            let threshold = Float::with_val(prec, prev / 3);
            if worst > threshold {
                stall_count += 1;
            } else {
                stall_count = 0;
            }
        }
        prev_worst = Some(worst.clone());
        if stall_count >= 2 {
            let can_escalate =
                !escalated && matches!(precond, Precond::F64(_)) && opt.allow_dd_escalation;
            if can_escalate {
                notes.push(format!(
                    "STALL at certified residual {:.3e} with the f64 preconditioner — escalating to double-double (the FP64 wall)",
                    worst.to_f64()
                ));
                // Fresh dd coarse inverse iteration, not just a dd solve of the
                // stalled state: the defect-correction fixed point carries a
                // non-kernel component ~σ_min(M_c)/σ_{k+1} which the coarse
                // inverse iteration (mixing floor ‖E_dd‖/gap) beats outright.
                let (cr, lu) = coarse_stage::<DdC, S>(src, k, opt.extra_coarse_vectors, opt.coarse_iters, prec);
                notes.push(cr.note.clone());
                if let Some(est) = cr.sigma_estimates.get(k) {
                    sigma_next_est = *est;
                }
                vs = cr.candidates;
                mgs_orthonormalize(&mut vs, prec, &mut notes);
                precond = Precond::Dd(lu);
                escalated = true;
                stall_count = 0;
                prev_worst = None;
                continue;
            } else {
                return Err(format!(
                    "refine_kernel: resource refusal — stalled at certified residual {:.3e} \
                     (target {}) after {iterations} iterations ({}); the target may sit below \
                     σ_k(M) or need a higher-precision preconditioner",
                    worst.to_f64(),
                    opt.target_residual_decimal,
                    if escalated {
                        "already dd-escalated"
                    } else if !opt.allow_dd_escalation {
                        "dd escalation disabled"
                    } else {
                        "dd preconditioner"
                    },
                ));
            }
        }
        // preconditioned correction: v ← v − M_c⁻¹(Mv), then re-orthonormalize
        for (v, r) in vs.iter_mut().zip(residuals.iter()) {
            let d = precond.correction(r, prec);
            for i in 0..n {
                v[i] = v[i].sub(&d[i]);
            }
        }
        mgs_orthonormalize(&mut vs, prec, &mut notes);
    }
    Err(format!(
        "refine_kernel: resource refusal — iteration budget {} exhausted without meeting the \
         certificate target {} (last worst bound {}); raise max_iters, raise the target, or \
         use a higher-precision coarse stage",
        opt.max_iters,
        opt.target_residual_decimal,
        prev_worst.map(|w| format!("{:.3e}", w.to_f64())).unwrap_or_else(|| "n/a".into()),
    ))
}

/// Refine and certify `k` kernel vectors of the in-memory hp matrix `m`.
pub fn refine_kernel(m: &MpMatrix, k: usize, opt: &KernelRefineOptions) -> Result<CertifiedKernel, String> {
    refine_core(m, k, opt)
}

/// Refine and certify `k` kernel vectors of an EXT-dumped matrix WITHOUT
/// materializing it in hp: all full-precision matvecs stream rows from disk
/// ([`ExtStream`]); only the coarse f64/dd copy (16/32 B per entry) is held in
/// memory. The working precision is `max(opt.prec_bits, exact_prec)` so every
/// certificate applies to the EXACT dumped matrix `Σ limbs`, not a re-rounding.
pub fn refine_kernel_streamed(ext_path: &str, k: usize, opt: &KernelRefineOptions) -> Result<CertifiedKernel, String> {
    let stream = ExtStream::open(ext_path).map_err(|e| format!("open {ext_path}: {e}"))?;
    let exact = stream.exact_prec().map_err(|e| format!("scan {ext_path}: {e}"))?;
    if exact > opt.prec_bits {
        let opt2 = KernelRefineOptions {
            prec_bits: exact,
            target_residual_decimal: opt.target_residual_decimal.clone(),
            max_iters: opt.max_iters,
            coarse: match &opt.coarse {
                CoarseStage::F64 => CoarseStage::F64,
                CoarseStage::DoubleDouble => CoarseStage::DoubleDouble,
                CoarseStage::External { candidates } => {
                    CoarseStage::External { candidates: candidates.clone() }
                }
            },
            coarse_iters: opt.coarse_iters,
            extra_coarse_vectors: opt.extra_coarse_vectors,
            with_gap: opt.with_gap,
            gap_prec_bits: opt.gap_prec_bits,
            allow_dd_escalation: opt.allow_dd_escalation,
            gap_shift_seed: opt.gap_shift_seed,
        };
        refine_core(&stream, k, &opt2)
    } else {
        refine_core(&stream, k, opt)
    }
}

// ===========================================================================
// Optional GPU coarse stage (cupy) — plumbing only, output NEVER trusted
// ===========================================================================

/// True iff `python3 -c "import cupy"` succeeds (runtime probe, never required).
pub fn cupy_available() -> bool {
    std::process::Command::new("python3")
        .args(["-c", "import cupy"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Parse the cupy helper's candidate dump. `expected_dim` is the TRUSTED
/// matrix dimension supplied by the caller; the dump text (header included) is
/// UNTRUSTED external-process output. Both header numbers are validated
/// against their trusted expectations BEFORE any allocation is sized from
/// them — a corrupted header (e.g. an absurd dim claim) must produce an
/// honest `Err`, never an attempted allocation.
fn parse_cupy_candidates(
    text: &str,
    k: usize,
    expected_dim: usize,
) -> Result<Vec<Vec<(String, String)>>, String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty cupy output")?;
    let mut hp = header.split_whitespace();
    let dim: usize = hp.next().and_then(|s| s.parse().ok()).ok_or("bad cupy header dim")?;
    let kk: usize = hp.next().and_then(|s| s.parse().ok()).ok_or("bad cupy header k")?;
    // VALIDATE BEFORE ALLOCATING: dim and kk come from the untrusted header.
    // Every collection below is sized only from the trusted k / expected_dim.
    if dim != expected_dim {
        return Err(format!(
            "cupy header claims dim {dim}, but the target matrix has dim {expected_dim} — \
             refusing the untrusted output"
        ));
    }
    if kk != k {
        return Err(format!("cupy helper returned {kk} vectors, asked for {k}"));
    }
    let mut candidates = Vec::with_capacity(k);
    for _ in 0..k {
        let mut v = Vec::with_capacity(expected_dim);
        for _ in 0..expected_dim {
            let line = lines.next().ok_or("truncated cupy output")?;
            let mut parts = line.split_whitespace();
            let re = parts.next().ok_or("missing re")?.to_string();
            let im = parts.next().ok_or("missing im")?.to_string();
            v.push((re, im));
        }
        candidates.push(v);
    }
    Ok(candidates)
}

/// Run `scripts/coarse_kernel_cupy.py` on a matrix dump (EXT limb format or
/// the raw-f64 format — the script distinguishes by size) and parse the k
/// candidate kernel vectors it writes as decimal strings. `expected_dim` is
/// the dimension of the target matrix; the helper's (untrusted) header must
/// match it exactly or the output is refused (see [`parse_cupy_candidates`]).
///
/// The result is UNTRUSTED INPUT by contract: feed it to
/// [`CoarseStage::External`], whose vectors are only starting points for the
/// hp refinement + certification. Nothing downstream believes the GPU.
pub fn coarse_kernel_via_cupy(
    matrix_path: &str,
    k: usize,
    expected_dim: usize,
) -> Result<Vec<Vec<(String, String)>>, String> {
    let script = format!("{}/scripts/coarse_kernel_cupy.py", env!("CARGO_MANIFEST_DIR"));
    if !std::path::Path::new(&script).exists() {
        return Err(format!("cupy helper script not found at {script}"));
    }
    let out_path = format!("{matrix_path}.cupy_candidates.txt");
    let output = std::process::Command::new("python3")
        .args([&script, matrix_path, &k.to_string(), &out_path])
        .output()
        .map_err(|e| format!("spawn python3: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cupy helper failed (status {:?}): {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = std::fs::read_to_string(&out_path).map_err(|e| format!("read {out_path}: {e}"))?;
    let _ = std::fs::remove_file(&out_path);
    parse_cupy_candidates(&text, k, expected_dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belyi::coset_graph::CosetGraph;
    use crate::belyi::modular_forms_hp::{assemble_scaled_ami, read_ext_matrix};
    use crate::belyi::mp_svd::{jacobi_svd, JacobiSvdOptions};
    use crate::belyi::solve::SolveParams;
    use crate::belyi::triangle_group::TriangleGroup;
    use crate::belyi::triangle_group_hp::TriangleGroupHp;

    const PREC: u32 = 256;

    // ---------------- deterministic test RNG ----------------
    struct Lcg(u64);
    impl Lcg {
        fn f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / (1u64 << 53) as f64 - 0.5
        }
    }

    fn tmp_path(name: &str) -> String {
        let mut p = std::env::current_exe().expect("current_exe");
        p.pop();
        p.push(format!("{name}.{}.bin", std::process::id()));
        p.to_str().expect("utf8 path").to_string()
    }

    // The EXT writer's limb split (verbatim algorithm), for building test dumps.
    fn split_scalar(x: &Float, nlimbs: usize, prec: u32, buf: &mut Vec<u8>) {
        let mut rem = Float::with_val(prec, x);
        for _ in 0..nlimbs {
            let hi = rem.to_f64();
            buf.extend_from_slice(&hi.to_le_bytes());
            rem = Float::with_val(prec, &rem - hi);
        }
    }

    fn write_ext(m: &MpMatrix, nlimbs: usize, path: &str) {
        let mut buf: Vec<u8> = Vec::with_capacity(5 + m.rows * m.cols * 2 * nlimbs * 8);
        buf.extend_from_slice(&(m.rows as u32).to_le_bytes());
        buf.push(nlimbs as u8);
        for e in &m.data {
            split_scalar(&e.re, nlimbs, m.prec, &mut buf);
            split_scalar(&e.im, nlimbs, m.prec, &mut buf);
        }
        std::fs::write(path, &buf).expect("write EXT test dump");
    }

    // ---------------- Householder-planted matrices ----------------
    // A = H(u1) H(u2) · diag(σ) · K(w1) K(w2), all reflectors exactly unitary in
    // exact arithmetic, so the singular values of A are the planted σ up to the
    // hp construction rounding (‖ΔA‖_F ≲ n·2^{1−prec}·σ_max ≈ 1e-73 at 256 bits,
    // n = 500 — far below every asserted scale). Right singular vector for
    // index j: v_j = K(w2)(K(w1) e_j).
    fn unit_vec(n: usize, rng: &mut Lcg, prec: u32) -> Vec<MpC> {
        let mut u: Vec<MpC> = (0..n).map(|_| MpC::from_f64(prec, rng.f64(), rng.f64())).collect();
        let nrm = norm2_hp(&u, prec);
        for z in u.iter_mut() {
            *z = z.div_real(&nrm);
        }
        u
    }

    // A ← (I − 2uu*) A, rayon over columns (deterministic: per-column folds).
    fn apply_left_reflector(a: &mut MpMatrix, u: &[MpC]) {
        let (n, prec) = (a.rows, a.prec);
        let two = Float::with_val(prec, 2);
        let cols: Vec<usize> = (0..a.cols).collect();
        let updates: Vec<Vec<MpC>> = cols
            .par_iter()
            .map(|&j| {
                let mut r = MpC::zero(prec);
                for i in 0..n {
                    r = r.add(&u[i].conj_mul(a.get(i, j)));
                }
                let r2 = r.scale(&two);
                (0..n).map(|i| a.get(i, j).sub(&u[i].mul(&r2))).collect()
            })
            .collect();
        for (j, col) in updates.into_iter().enumerate() {
            for (i, z) in col.into_iter().enumerate() {
                a.set(i, j, z);
            }
        }
    }

    // A ← A (I − 2ww*), rayon over rows.
    fn apply_right_reflector(a: &mut MpMatrix, w: &[MpC]) {
        let (n, prec) = (a.rows, a.prec);
        let two = Float::with_val(prec, 2);
        let rows: Vec<usize> = (0..n).collect();
        let updates: Vec<Vec<MpC>> = rows
            .par_iter()
            .map(|&i| {
                let mut c = MpC::zero(prec);
                for j in 0..a.cols {
                    c = c.add(&a.get(i, j).mul(&w[j]));
                }
                let c2 = c.scale(&two);
                (0..a.cols).map(|j| a.get(i, j).sub(&c2.mul(&w[j].conj()))).collect()
            })
            .collect();
        for (i, row) in updates.into_iter().enumerate() {
            for (j, z) in row.into_iter().enumerate() {
                a.set(i, j, z);
            }
        }
    }

    fn apply_reflector_to_vec(v: &mut Vec<MpC>, w: &[MpC], prec: u32) {
        let two = Float::with_val(prec, 2);
        let mut r = MpC::zero(prec);
        for i in 0..v.len() {
            r = r.add(&w[i].conj_mul(&v[i]));
        }
        let r2 = r.scale(&two);
        for i in 0..v.len() {
            v[i] = v[i].sub(&w[i].mul(&r2));
        }
    }

    /// Planted matrix + the exact right singular vectors for indices `want`.
    fn planted(n: usize, sigmas: &[f64], want: &[usize], prec: u32, seed: u64) -> (MpMatrix, Vec<Vec<MpC>>) {
        assert_eq!(sigmas.len(), n);
        let mut rng = Lcg(seed);
        let u1 = unit_vec(n, &mut rng, prec);
        let u2 = unit_vec(n, &mut rng, prec);
        let w1 = unit_vec(n, &mut rng, prec);
        let w2 = unit_vec(n, &mut rng, prec);
        let mut a = MpMatrix::zeros(n, n, prec);
        for (j, s) in sigmas.iter().enumerate() {
            a.set(j, j, MpC::from_f64(prec, *s, 0.0));
        }
        apply_left_reflector(&mut a, &u2);
        apply_left_reflector(&mut a, &u1);
        apply_right_reflector(&mut a, &w1);
        apply_right_reflector(&mut a, &w2);
        let vs = want
            .iter()
            .map(|&j| {
                let mut v = vec![MpC::zero(prec); n];
                v[j] = MpC::one(prec);
                apply_reflector_to_vec(&mut v, &w1, prec);
                apply_reflector_to_vec(&mut v, &w2, prec);
                v
            })
            .collect();
        (a, vs)
    }

    /// ‖(I − BB*)x‖ for an orthonormal basis B (columns) — the projection
    /// residual of x against span(B).
    fn projection_residual(basis: &[Vec<MpC>], x: &[MpC], prec: u32) -> f64 {
        let n = x.len();
        let coefs: Vec<MpC> = basis.iter().map(|b| dot_conj_hp(b, x, prec)).collect();
        let mut resid2 = Float::with_val(prec, 0);
        for i in 0..n {
            let mut p = MpC::zero(prec);
            for (b, c) in basis.iter().zip(coefs.iter()) {
                p = p.add(&b[i].mul(c));
            }
            resid2 += x[i].sub(&p).abs2();
        }
        resid2.to_f64().sqrt()
    }

    // =======================================================================
    // EFT unit tests
    // =======================================================================

    // two_sum: s + e == a + b EXACTLY, verified in MPFR at a precision wide
    // enough to hold both exponent ranges (adversarial pairs included).
    #[test]
    fn eft_two_sum_exact() {
        let pairs: [(f64, f64); 8] = [
            (1.0, 2f64.powi(-60)),
            (2f64.powi(53), 1.0),
            (0.1, 0.2),
            (-1.0, 1e-300),
            (2f64.powi(-1000), 2f64.powi(-1050)),
            (1e16, -1.0),
            (3.141592653589793, 2.718281828459045e-12),
            (-2f64.powi(52), 2f64.powi(-52)),
        ];
        const P: u32 = 2400; // covers the widest span (2^53 … 2^-1050) exactly
        for &(a, b) in &pairs {
            let (s, e) = two_sum(a, b);
            let lhs = Float::with_val(P, s) + Float::with_val(P, e);
            let rhs = Float::with_val(P, a) + Float::with_val(P, b);
            assert_eq!(lhs, rhs, "two_sum not exact for ({a:e}, {b:e})");
            // and s must be the correctly rounded sum
            assert_eq!(s, a + b, "two_sum hi is not RN(a+b) for ({a:e}, {b:e})");
            let (qs, qe) = two_sum(b, a); // symmetry of the Knuth variant
            let qlhs = Float::with_val(P, qs) + Float::with_val(P, qe);
            assert_eq!(qlhs, rhs, "two_sum not exact under swap for ({a:e}, {b:e})");
        }
    }

    // two_prod (FMA): p + e == a·b EXACTLY when the product and its error stay
    // in the normal range (the documented precondition; exact f64 products fit
    // in 106 bits, so MPFR at 120 bits is an exact reference).
    #[test]
    fn eft_two_prod_exact() {
        let pairs: [(f64, f64); 7] = [
            (1.0 + 2f64.powi(-52), 1.0 - 2f64.powi(-52)),
            (3.141592653589793, 2.718281828459045),
            (2f64.powi(500), 2f64.powi(-400)),
            (3.14e150, 2.71e-140),
            (-1.0000000000000002, 1.0000000000000002),
            (0.1, 0.3),
            (2f64.powi(-400), 2f64.powi(-401)),
        ];
        const P: u32 = 120;
        for &(a, b) in &pairs {
            let (p, e) = two_prod(a, b);
            let lhs = Float::with_val(P, p) + Float::with_val(P, e);
            let rhs = Float::with_val(P, a) * Float::with_val(P, b);
            assert_eq!(lhs, rhs, "two_prod not exact for ({a:e}, {b:e})");
            assert_eq!(p, a * b, "two_prod hi is not RN(a·b) for ({a:e}, {b:e})");
        }
    }

    // dd arithmetic against a 256-bit MPFR reference: single ops near the dd
    // roundoff 2^-105, and a length-200 complex inner product to ~2^-100
    // (benign same-sign accumulation; bounds asserted at measured + margin).
    #[test]
    fn dd_arithmetic_and_inner_product_vs_mpfr() {
        let mut rng = Lcg(0xfeed_beef_dead_cafe);
        let rand_float = |rng: &mut Lcg| -> Float {
            // uniform-ish in [0.5, 1.5]: benign scale, full 106-bit significands
            Float::with_val(PREC, 1.0 + rng.f64())
        };
        let rel_err = |dd: Dd, reference: &Float| -> f64 {
            let diff = Float::with_val(PREC, dd.to_float(PREC) - reference);
            let denom = Float::with_val(PREC, reference.clone().abs());
            (Float::with_val(PREC, diff.abs() / denom)).to_f64()
        };
        let mut worst_op = 0f64;
        for _ in 0..200 {
            let fa = rand_float(&mut rng);
            let fb = rand_float(&mut rng);
            let (a, b) = (Dd::from_float(&fa), Dd::from_float(&fb));
            // compare against the op applied to the dd-ROUNDED inputs (isolates
            // the op error from the input rounding)
            let (ra, rb) = (a.to_float(PREC), b.to_float(PREC));
            worst_op = worst_op.max(rel_err(a.add(b), &Float::with_val(PREC, &ra + &rb)));
            worst_op = worst_op.max(rel_err(a.sub(b), &Float::with_val(PREC, &ra - &rb)));
            worst_op = worst_op.max(rel_err(a.mul(b), &Float::with_val(PREC, &ra * &rb)));
            worst_op = worst_op.max(rel_err(a.div(b), &Float::with_val(PREC, &ra / &rb)));
            worst_op = worst_op.max(rel_err(a.sqrt(), &ra.clone().sqrt()));
        }
        eprintln!("[dd] worst single-op relative error {worst_op:.3e} (2^-102 = {:.3e})", 2f64.powi(-102));
        assert!(worst_op < 2f64.powi(-102), "dd single-op error too large: {worst_op:.3e}");

        // complex inner product, length 200
        let n = 200usize;
        let av: Vec<(Float, Float)> = (0..n).map(|_| (rand_float(&mut rng), rand_float(&mut rng))).collect();
        let bv: Vec<(Float, Float)> = (0..n).map(|_| (rand_float(&mut rng), rand_float(&mut rng))).collect();
        let mut acc_dd = DdC { re: Dd::ZERO, im: Dd::ZERO };
        let mut acc_ref = MpC::zero(PREC);
        for ((ar, ai), (br, bi)) in av.iter().zip(bv.iter()) {
            let za = DdC { re: Dd::from_float(ar), im: Dd::from_float(ai) };
            let zb = DdC { re: Dd::from_float(br), im: Dd::from_float(bi) };
            acc_dd = acc_dd.add(za.conj().mul(zb));
            // reference on the SAME dd-rounded inputs at 256 bits
            let ma = MpC::new(za.re.to_float(PREC), za.im.to_float(PREC));
            let mb = MpC::new(zb.re.to_float(PREC), zb.im.to_float(PREC));
            acc_ref = acc_ref.add(&ma.conj_mul(&mb));
        }
        let dre = Float::with_val(PREC, acc_dd.re.to_float(PREC) - &acc_ref.re);
        let dim_ = Float::with_val(PREC, acc_dd.im.to_float(PREC) - &acc_ref.im);
        let diff = Float::with_val(PREC, dre.clone() * &dre + dim_.clone() * &dim_).sqrt();
        let scale = acc_ref.abs(PREC);
        let rel = Float::with_val(PREC, diff / scale).to_f64();
        eprintln!("[dd] length-{n} complex inner product vs 256-bit MPFR: rel err {rel:.3e} (2^-100 = {:.3e})", 2f64.powi(-100));
        assert!(rel < 2f64.powi(-98), "dd inner product error too large: {rel:.3e}");
    }

    // =======================================================================
    // Coarse LU
    // =======================================================================

    #[test]
    fn coarse_lu_solve_and_conj_transpose_solve() {
        let n = 30usize;
        let mut rng = Lcg(0x0123_4567_89ab_cdef);
        // diagonally dominant complex matrix at 256 bits, plus rhs
        let mut m = MpMatrix::zeros(n, n, PREC);
        for i in 0..n {
            for j in 0..n {
                let boost = if i == j { 6.0 } else { 0.0 };
                m.set(i, j, MpC::from_f64(PREC, rng.f64() + boost, rng.f64()));
            }
        }
        let b: Vec<MpC> = (0..n).map(|_| MpC::from_f64(PREC, rng.f64(), rng.f64())).collect();

        fn check<T: CoarseScalar + PartialEq + std::fmt::Debug>(m: &MpMatrix, b: &[MpC], tol: f64) {
            let n = m.rows;
            let mc: Vec<T> = coarse_copy(m, m.prec);
            let lu = lu_factor(mc.clone(), n);
            assert!(!lu.had_zero_pivot);
            // determinism policy check: the rayon-threaded elimination must be
            // bitwise reproducible (each row is one task's sequential update).
            let lu2 = lu_factor(mc.clone(), n);
            assert!(lu.a == lu2.a && lu.piv == lu2.piv, "{} LU not bitwise deterministic", T::NAME);
            // A x = b
            let mut x: Vec<T> = b.iter().map(T::from_mpc).collect();
            lu.solve(&mut x);
            let mut worst = 0f64;
            for i in 0..n {
                let mut acc = T::zero();
                for j in 0..n {
                    acc = acc.add(mc[i * n + j].mul(x[j]));
                }
                let r = acc.sub(T::from_mpc(&b[i]));
                worst = worst.max(r.abs2_f64().sqrt());
            }
            eprintln!("[lu:{}] ‖Ax−b‖∞ = {worst:.3e}", T::NAME);
            assert!(worst < tol, "{} LU solve residual {worst:.3e} ≥ {tol:.1e}", T::NAME);
            // A* x = b
            let mut y: Vec<T> = b.iter().map(T::from_mpc).collect();
            lu.solve_conj_transpose(&mut y);
            let mut worst_t = 0f64;
            for i in 0..n {
                let mut acc = T::zero();
                for j in 0..n {
                    acc = acc.add(mc[j * n + i].conj().mul(y[j]));
                }
                let r = acc.sub(T::from_mpc(&b[i]));
                worst_t = worst_t.max(r.abs2_f64().sqrt());
            }
            eprintln!("[lu:{}] ‖A*x−b‖∞ = {worst_t:.3e}", T::NAME);
            assert!(worst_t < tol, "{} LU conj-transpose residual {worst_t:.3e} ≥ {tol:.1e}", T::NAME);
        }
        check::<Complex64>(&m, &b, 1e-12);
        check::<DdC>(&m, &b, 1e-27);
    }

    // =======================================================================
    // Streamed matvec: bitwise determinism + bitwise agreement with the
    // materialized path (the stated rayon-determinism policy, tested).
    // =======================================================================

    #[test]
    fn streamed_matvec_bitwise_matches_materialized() {
        let n = 40usize;
        let prec: u32 = 212; // = 4·53: the nlimbs=4 split is lossless
        let nlimbs = 4usize;
        let mut rng = Lcg(0xabcdef012345);
        let mut m = MpMatrix::zeros(n, n, prec);
        for i in 0..n {
            for j in 0..n {
                // spread magnitudes over ~120 dB to exercise the limb split
                let mag = 10f64.powi((rng.f64() * 12.0) as i32);
                m.set(i, j, MpC::from_f64(prec, rng.f64() * mag, rng.f64() * mag));
            }
        }
        let path = tmp_path("kernel_refine_ext_stream");
        write_ext(&m, nlimbs, &path);

        let stream = ExtStream::open(&path).expect("open");
        assert_eq!((stream.dim, stream.nlimbs), (n, nlimbs));
        let exact = stream.exact_prec().expect("scan");
        let m_disk = read_ext_matrix(&path).expect("read");
        assert_eq!(m_disk.prec, exact, "ExtStream and read_ext_matrix derive the same exact precision");

        // rows bitwise identical to the materialized matrix
        for i in 0..n {
            let row = RowSource::row(&stream, i, exact);
            for j in 0..n {
                let e = m_disk.get(i, j);
                assert!(row[j].re == e.re && row[j].im == e.im, "row {i} entry {j} differs");
            }
        }

        let v: Vec<MpC> = (0..n).map(|_| MpC::from_f64(exact, rng.f64(), rng.f64())).collect();
        let s1 = stream.matvec(&v, exact).expect("matvec 1");
        let s2 = stream.matvec(&v, exact).expect("matvec 2");
        let s3 = matvec_hp(&m_disk, &v, exact);
        let s4 = matvec_hp(&m_disk, &v, exact);
        for i in 0..n {
            assert!(s1[i].re == s2[i].re && s1[i].im == s2[i].im, "streamed matvec not run-to-run bitwise stable");
            assert!(s3[i].re == s4[i].re && s3[i].im == s4[i].im, "materialized matvec not run-to-run bitwise stable");
            assert!(s1[i].re == s3[i].re && s1[i].im == s3[i].im, "streamed vs materialized matvec differ bitwise at row {i}");
        }
        std::fs::remove_file(&path).expect("cleanup");
    }

    // =======================================================================
    // GOLDEN: the (5,3,3) k=6 kernel (dim S_6 = 3) — refined subspace matches
    // the hp Jacobi SVD kernel, certificates below the SolveParams threshold.
    // =======================================================================

    #[test]
    fn golden_5_3_3_kernel_matches_jacobi_svd() {
        let (k_weight, n, q) = (6i64, 48usize, 96usize);
        let tg64 = TriangleGroup::new(5, 3, 3);
        let tg = TriangleGroupHp::new(5, 3, 3, PREC);
        let s0 = vec![4, 0, 1, 2, 3];
        let s1 = vec![1, 2, 0, 3, 4];
        let mut cg = CosetGraph::build(&tg64, &s0, &s1);
        cg.compactify(&tg64);
        let dim = n + 1;
        let (a, _rho) = assemble_scaled_ami(&tg64, &tg, &cg, k_weight, n, q, 1.0, &tg.z_a);
        let mut data = Vec::with_capacity(dim * dim);
        for row in &a {
            for z in row {
                data.push(MpC::new(Float::with_val(PREC, z.real()), Float::with_val(PREC, z.imag())));
            }
        }
        let m = MpMatrix::from_row_major(dim, dim, PREC, data).expect("square");

        // SolveParams-derived threshold: ρ^48 ≈ 5e-14 ⇒ digits = 13 ⇒ "1e-8"
        let sp = SolveParams::new(PREC, n, 13).expect("params");
        assert_eq!(sp.threshold_decimal, "1e-8");

        let opt = KernelRefineOptions::new(PREC, &sp.threshold_decimal);
        let ck = refine_kernel(&m, 3, &opt).expect("refinement must converge on the golden case");
        let threshold = Float::with_val(PREC, Float::parse(&sp.threshold_decimal).unwrap());
        for (j, b) in ck.residual_bounds.iter().enumerate() {
            eprintln!("[golden] certified ‖Mv_{j}‖/‖v_{j}‖ ≤ {:.3e}", b.to_f64());
            assert!(*b <= threshold, "certificate {j} above the SolveParams threshold");
        }

        // independent verification of the certificate at a DIFFERENT precision
        // (320 bits): the directly computed residual must sit under the bound.
        for (j, v) in ck.vectors.iter().enumerate() {
            let v320: Vec<MpC> = v.iter().map(|z| MpC::new(Float::with_val(320, &z.re), Float::with_val(320, &z.im))).collect();
            let s = matvec_hp(&m, &v320, 320);
            let ratio = Float::with_val(320, norm2_hp(&s, 320) / norm2_hp(&v320, 320));
            assert!(
                Float::with_val(PREC, &ratio) <= ck.residual_bounds[j],
                "320-bit recomputed residual {:.3e} exceeds the certificate {:.3e}",
                ratio.to_f64(),
                ck.residual_bounds[j].to_f64()
            );
        }

        // kernel SUBSPACE comparison against the hp Jacobi SVD (projection
        // residual — per-vector comparison is ill-conditioned inside the
        // near-degenerate kernel cluster, per the A8/mp_svd docs).
        let svd = jacobi_svd(&m, &JacobiSvdOptions::new(PREC, 80, "1e-70", "1e-40")).expect("svd");
        let ker = svd.right_nullspace_basis(&threshold);
        assert_eq!(ker.cols, 3, "dim S_6 = 3");
        let svd_basis: Vec<Vec<MpC>> = (0..ker.cols)
            .map(|j| (0..dim).map(|i| ker.get(i, j).clone()).collect())
            .collect();
        let mut worst = 0f64;
        for v in &ck.vectors {
            worst = worst.max(projection_residual(&svd_basis, v, PREC));
        }
        eprintln!("[golden] worst projection residual of refined kernel onto SVD kernel: {worst:.3e}");
        assert!(worst < 1e-6, "refined kernel subspace diverges from the Jacobi SVD kernel: {worst:.3e}");

        // gap certificate: rigorous σ_3 upper below the threshold, rigorous
        // σ_4 lower above it, separation certified.
        let gap = ck.gap.as_ref().expect("gap requested");
        eprintln!(
            "[golden] gap: σ_3 ≤ {:.3e}, σ_4 ≥ {:.3e} (heuristic σ̂_4 ≈ {:.3e}), separated = {}",
            gap.sigma_k_upper.to_f64(),
            gap.sigma_next_lower.to_f64(),
            gap.sigma_next_estimate_heuristic,
            gap.separated
        );
        assert!(gap.sigma_k_upper <= threshold, "σ_3 upper bound above threshold");
        assert!(gap.sigma_next_lower > threshold, "σ_4 lower bound must clear the threshold");
        assert!(gap.separated, "gap must be certified separated");
        eprintln!("[golden] provenance: {}", ck.provenance);
    }

    // =======================================================================
    // Planted kernel at dim 500: known null vectors by construction, planted
    // spectrum for the gap certificate to be checked against.
    // =======================================================================

    #[test]
    fn planted_kernel_dim500_certified() {
        let n = 500usize;
        let k = 3usize;
        let mut sigmas = vec![0.0; n];
        // σ_4 = 0.3 exactly one; the rest spread in [0.35, 2.0]
        sigmas[3] = 0.3;
        for (t, s) in sigmas.iter_mut().enumerate().skip(4) {
            *s = 0.35 + 1.65 * ((t - 4) as f64) / ((n - 5) as f64);
        }
        let (m, planted_kernel) = planted(n, &sigmas, &[0, 1, 2], PREC, 0x5eed_0001);

        // Target 1e-25: the planted kernel σ are exact zeros up to the ~1e-73
        // construction rounding, and the kernel here is EXACT, so the f64
        // defect correction converges quadratically past its own ε (measured:
        // 1.56e-27 after 2 iterations, ~13 digits/iteration) — no escalation
        // is needed on this shape (the FP64 wall needs sub-ε *separation*,
        // exercised in fp64_wall_dd_fallback).
        let mut opt = KernelRefineOptions::new(PREC, "1e-25");
        opt.extra_coarse_vectors = 1;
        let t0 = std::time::Instant::now();
        let ck = refine_kernel(&m, k, &opt).expect("planted refinement must converge");
        eprintln!("[planted500] refine+certify took {:.1?} ({} iterations)", t0.elapsed(), ck.iterations);

        let target = Float::with_val(PREC, Float::parse("1e-25").unwrap());
        for (j, b) in ck.residual_bounds.iter().enumerate() {
            eprintln!("[planted500] certified residual {j}: {:.3e}", b.to_f64());
            assert!(*b <= target);
        }
        // subspace vs the PLANTED kernel (independent of any SVD): mixing floor
        // ~‖E_dd‖/gap ≈ 1e-29 here; asserted with margin.
        let mut worst = 0f64;
        for v in &ck.vectors {
            worst = worst.max(projection_residual(&planted_kernel, v, PREC));
        }
        eprintln!("[planted500] worst projection residual onto planted kernel: {worst:.3e}");
        assert!(worst < 1e-24, "refined subspace diverges from the planted kernel: {worst:.3e}");

        // gap certificate vs the planted spectrum: σ_4 = 0.3. The lower bound
        // must be RIGOROUS (≤ true σ_4 + construction slack) and useful (≥ 0.1).
        let gap = ck.gap.as_ref().expect("gap");
        let lower = gap.sigma_next_lower.to_f64();
        eprintln!(
            "[planted500] gap: σ_3 ≤ {:.3e}, σ_4 ≥ {lower:.6}, σ̂_4(heuristic) ≈ {:.4}, separated = {}",
            gap.sigma_k_upper.to_f64(),
            gap.sigma_next_estimate_heuristic,
            gap.separated
        );
        assert!(gap.sigma_k_upper.to_f64() < 1e-24, "σ_3 upper bound above the certified target");
        assert!(gap.separated);
        assert!(lower >= 0.1, "σ_4 lower bound too weak: {lower:.3e}");
        assert!(lower <= 0.3 + 1e-9, "σ_4 lower bound EXCEEDS the planted σ_4 — unsound: {lower}");
    }

    // =======================================================================
    // The FP64 wall: kernel σ = 1e-20 under a second small σ = 1e-17. f64
    // cannot separate them (both below the f64 matrix-rounding floor; the f64
    // preconditioner amplifies both equally) — with dd escalation disabled the
    // refinement must REFUSE. The dd coarse stage separates and certifies.
    // =======================================================================

    #[test]
    fn fp64_wall_dd_fallback() {
        let n = 80usize;
        let mut sigmas = vec![0.0; n];
        sigmas[0] = 1e-20;
        sigmas[1] = 1e-17;
        for (t, s) in sigmas.iter_mut().enumerate().skip(2) {
            *s = 0.5 + 1.5 * ((t - 2) as f64) / ((n - 3) as f64);
        }
        let (m, planted_kernel) = planted(n, &sigmas, &[0], PREC, 0x5eed_0002);

        // (a) f64 coarse, escalation disabled: must refuse (Err), never emit a
        // certificate it cannot meet.
        let mut opt_f64 = KernelRefineOptions::new(PREC, "3e-20");
        opt_f64.allow_dd_escalation = false;
        opt_f64.max_iters = 12;
        opt_f64.with_gap = false;
        let r = refine_kernel(&m, 1, &opt_f64);
        match r {
            Err(e) => eprintln!("[wall] f64-only refinement honestly refused: {e}"),
            Ok(ck) => {
                // if f64 DID reach the target the certificate must still be
                // honest — verify independently, and the subspace must be the
                // true kernel. (Empirically it stalls; this branch guards
                // against a silent wrong-subspace 'success'.)
                let worst = projection_residual(&planted_kernel, &ck.vectors[0], PREC);
                panic!(
                    "f64-only path unexpectedly claimed success (projection residual {worst:.3e}) — \
                     the wall construction no longer walls; tighten σ₂"
                );
            }
        }

        // (b) dd coarse: must converge and certify at the target.
        let mut opt_dd = KernelRefineOptions::new(PREC, "3e-20");
        opt_dd.coarse = CoarseStage::DoubleDouble;
        let ck = refine_kernel(&m, 1, &opt_dd).expect("dd coarse stage must clear the FP64 wall");
        eprintln!("[wall] dd certified residual: {:.3e}", ck.residual_bounds[0].to_f64());
        let worst = projection_residual(&planted_kernel, &ck.vectors[0], PREC);
        eprintln!("[wall] dd projection residual onto planted kernel: {worst:.3e}");
        assert!(worst < 1e-2, "dd found the wrong kernel direction: {worst:.3e}");
        let gap = ck.gap.as_ref().expect("gap");
        let lower = gap.sigma_next_lower.to_f64();
        eprintln!(
            "[wall] gap: σ_1 ≤ {:.3e}, σ_2 ≥ {lower:.3e}, separated = {}",
            gap.sigma_k_upper.to_f64(),
            gap.separated
        );
        assert!(gap.separated, "the 1e-20 vs 1e-17 gap must be certified");
        assert!(lower <= 1.0000001e-17, "σ_2 lower bound exceeds the planted σ_2 — unsound");
        assert!(lower > 3e-20, "σ_2 lower bound must clear the kernel certificate");

        // (c) f64 coarse WITH escalation allowed: the automatic fallback must
        // rescue the run and note the stall in provenance.
        let mut opt_esc = KernelRefineOptions::new(PREC, "3e-20");
        opt_esc.with_gap = false;
        let ck2 = refine_kernel(&m, 1, &opt_esc).expect("auto-escalation must rescue the f64 start");
        assert!(
            ck2.provenance.contains("escalating to double-double"),
            "provenance must record the FP64-wall escalation: {}",
            ck2.provenance
        );
        let worst2 = projection_residual(&planted_kernel, &ck2.vectors[0], PREC);
        assert!(worst2 < 1e-2, "escalated run found the wrong kernel: {worst2:.3e}");
    }

    // =======================================================================
    // The auditor's M1 shape: EVERY deflated σ sits below the coarse epsilon
    // (here: exact planted kernel, k = 2), so the coarse block inverse
    // iteration collapses all k+1 vectors into the kernel's noise subspace and
    // the MGS leaves the (k+1)-th as cancellation garbage — the heuristic
    // σ̂_{k+1} lands orders of magnitude above the true σ_3 = 1e-3 (measured
    // 2.4e-2 emergent here; the auditor measured 0.69 on the production
    // shape). The OLD 8-rung hp ladder from (σ̂/2)² covered only ~2.4 decades
    // of shift below the seed: with the audited σ̂ = 0.69 its lowest rung
    // (0.345)²/4⁷ ≈ 7.3e-6 sat ABOVE the true rung λ_min = σ_3² = 1e-6, so
    // every attempt failed and the certificate came back vacuous
    // (sigma_next_lower = 0, separated = false) despite a >20-decade true
    // gap. The dd locator must now find the rung independently of the seed.
    // =======================================================================

    #[test]
    fn gap_locator_rescues_destroyed_heuristic_m1() {
        let n = 60usize;
        let k = 2usize;
        let mut sigmas = vec![0.0; n];
        sigmas[2] = 1e-3; // σ_{k+1}: >20 decades above the exact planted kernel
        for (t, s) in sigmas.iter_mut().enumerate().skip(3) {
            *s = 0.5 + 1.5 * ((t - 3) as f64) / ((n - 4) as f64);
        }
        let (m, planted_kernel) = planted(n, &sigmas, &[0, 1], PREC, 0x5eed_0005);

        // --- end-to-end: the production sequence (refine, then gap) must now
        // come back separated regardless of the emergent heuristic's quality.
        let opt = KernelRefineOptions::new(PREC, "1e-25");
        let ck = refine_kernel(&m, k, &opt).expect("M1 refinement must converge");
        let mut worst = 0f64;
        for v in &ck.vectors {
            worst = worst.max(projection_residual(&planted_kernel, v, PREC));
        }
        assert!(worst < 1e-20, "M1 refinement found the wrong kernel: {worst:.3e}");
        let gap = ck.gap.as_ref().expect("gap requested");
        eprintln!(
            "[m1] emergent heuristic σ̂_3 = {:.3e} (true σ_3 = 1e-3); end-to-end note: {}",
            gap.sigma_next_estimate_heuristic, gap.note
        );
        assert!(gap.separated, "end-to-end M1 gap must be certified separated");
        assert!(gap.sigma_next_lower.to_f64() >= 2e-4 && gap.sigma_next_lower.to_f64() <= 1.0000001e-3);

        // --- the PINNED audited failure: the auditor's destroyed seed value
        // σ̂ = 0.69 against true σ_3 = 1e-3, on the refined vectors. Under the
        // old code this is PROVABLY vacuous: the 8 rungs (0.345)²·4^{-a},
        // a = 0..7, are all ≥ 7.27e-6 > λ_min + β, so every Cholesky attempt
        // fails and sigma_next_lower stays 0. The locator must ignore the
        // seed and find the rung.
        let destroyed_seed = 0.69;
        let old_floor = (destroyed_seed / 2.0) * (destroyed_seed / 2.0) / 4f64.powi(7);
        assert!(
            old_floor > 4e-6,
            "arithmetic sanity: the audited seed's old-ladder floor must sit above λ_min = 1e-6"
        );
        let g2 = gap_certificate(&m, &ck.vectors, PREC, PREC, destroyed_seed);
        assert!(g2.note.contains("dd locator: rung"), "note must record the locator hit: {}", g2.note);
        let lower = g2.sigma_next_lower.to_f64();
        let upper = g2.sigma_k_upper.to_f64();
        assert!(lower > 0.0, "vacuous certificate: sigma_next_lower = 0 with an obtainable rung");
        assert!(
            lower >= 2e-4,
            "located rung too weak: σ_3 ≥ {lower:.3e} (expected within ~2 quarterings of 1e-3)"
        );
        assert!(lower <= 1.0000001e-3, "σ_3 lower bound EXCEEDS the planted σ_3 — unsound: {lower:.6e}");
        assert!(upper < 1e-24, "σ_2 upper bound above the certified target: {upper:.3e}");
        assert!(g2.separated, "the certified gap must be separated despite the destroyed seed");
        let decades = (lower / upper).log10();
        eprintln!(
            "[m1] pinned destroyed seed 0.69 (old ladder floor {old_floor:.3e} > 1e-6): \
             certified σ_2 ≤ {upper:.3e}, σ_3 ≥ {lower:.3e} — {decades:.1} decades of separation"
        );
        eprintln!("[m1] pinned note: {}", g2.note);
        assert!(decades > 20.0, "certified separation too narrow: {decades:.1} decades");
    }

    // =======================================================================
    // gap_shift_seed plumbing + the seed-ladder fallback: a rung BELOW the dd
    // locator floor (σ_{k+1} = 1e-17 ⇒ λ_min(H) = 1e-34 ≪ ‖H‖_F·2⁻¹⁰⁰) is
    // findable only from the seed. A garbage seed yields the labeled
    // no-certificate outcome (honest, never unsound); the right seed yields
    // the separated certificate. And the KernelRefineOptions::gap_shift_seed
    // override must reach the certificate verbatim.
    // =======================================================================

    #[test]
    fn gap_shift_seed_fallback_below_dd_floor_and_plumbing() {
        let n = 60usize;
        let mut sigmas = vec![0.0; n];
        sigmas[1] = 1e-17;
        for (t, s) in sigmas.iter_mut().enumerate().skip(2) {
            *s = 0.5 + 1.5 * ((t - 2) as f64) / ((n - 3) as f64);
        }
        let (m, planted_kernel) = planted(n, &sigmas, &[0], PREC, 0x5eed_0006);

        // (a) garbage seed, rung below the dd floor: both stages exhaust and
        // the outcome is the LABELED no-certificate (sigma_next_lower = 0) —
        // honest, and the note records the genuine locator search.
        let bad = gap_certificate(&m, &planted_kernel, PREC, PREC, 0.7);
        eprintln!("[seed] garbage seed note: {}", bad.note);
        assert!(bad.note.contains("NO rung above the dd floor"), "locator must record its exhausted search");
        assert!(bad.note.contains("no shift verified"), "no-certificate outcome must stay labeled");
        assert_eq!(bad.sigma_next_lower.to_f64(), 0.0);
        assert!(!bad.separated);

        // (b) the right seed rescues via the fallback ladder: shift (σ̂/2)² =
        // 2.5e-35 < λ_min(H) = 1e-34 verifies, giving σ_2 ≥ ~5e-18.
        let good = gap_certificate(&m, &planted_kernel, PREC, PREC, 1e-17);
        eprintln!("[seed] good seed note: {}", good.note);
        let lower = good.sigma_next_lower.to_f64();
        assert!(good.separated, "an obtainable certificate must be found from the right seed");
        assert!(lower > 3e-18, "fallback rung too weak: {lower:.3e}");
        assert!(lower <= 1.0000001e-17, "σ_2 lower bound exceeds the planted σ_2 — unsound: {lower:.3e}");

        // (c) end-to-end plumbing: gap_shift_seed overrides the coarse
        // heuristic, is recorded verbatim in the certificate, and the override
        // is noted in provenance. (The dd escalation's own σ̂_2 ≈ 1e-17 would
        // also work here — the exact recorded value proves the override won.)
        let mut opt = KernelRefineOptions::new(PREC, "3e-20");
        opt.coarse = CoarseStage::DoubleDouble;
        opt.gap_shift_seed = Some(2e-17);
        let ck = refine_kernel(&m, 1, &opt).expect("dd coarse + seeded gap must converge");
        assert!(
            ck.provenance.contains("gap shift seed OVERRIDDEN by caller"),
            "provenance must record the seed override: {}",
            ck.provenance
        );
        let gap = ck.gap.as_ref().expect("gap");
        assert_eq!(
            gap.sigma_next_estimate_heuristic, 2e-17,
            "gap_shift_seed must reach the certificate verbatim"
        );
        assert!(gap.separated, "seeded production-style run must certify the gap");
        let l2 = gap.sigma_next_lower.to_f64();
        assert!(l2 > 3e-18 && l2 <= 1.0000001e-17, "seeded lower bound out of range: {l2:.3e}");
    }

    // =======================================================================
    // Corrupted cupy header: untrusted header numbers must be validated
    // BEFORE any allocation is sized from them. A huge dim claim must come
    // back as an honest Err — this test completing at all is the evidence
    // that no usize::MAX-sized allocation was attempted.
    // =======================================================================

    #[test]
    fn cupy_corrupted_header_refused_without_allocation() {
        // huge dim claim (the OOM class this project keeps meeting)
        let huge = format!("{} 2\nnot even data\n", usize::MAX);
        let e = parse_cupy_candidates(&huge, 2, 40).unwrap_err();
        assert!(e.contains("refusing"), "huge dim claim must be refused: {e}");

        // modest but wrong dim claim
        let wrong = "39 2\n0 0\n";
        let e = parse_cupy_candidates(wrong, 2, 40).unwrap_err();
        assert!(e.contains("claims dim 39"), "dim mismatch must be refused: {e}");

        // k mismatch
        let badk = "40 3\n0 0\n";
        let e = parse_cupy_candidates(badk, 2, 40).unwrap_err();
        assert!(e.contains("asked for 2"), "k mismatch must be refused: {e}");

        // well-formed header, truncated body
        let trunc = "2 1\n0.5 0.25\n";
        let e = parse_cupy_candidates(trunc, 1, 2).unwrap_err();
        assert!(e.contains("truncated"), "truncated body must be refused: {e}");

        // empty and garbage headers
        assert!(parse_cupy_candidates("", 1, 2).is_err());
        assert!(parse_cupy_candidates("x y\n", 1, 2).is_err());

        // the valid case still parses
        let ok = parse_cupy_candidates("2 1\n0.5 0.25\n-1 3e-2\n", 1, 2).expect("valid dump");
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].len(), 2);
        assert_eq!(ok[0][1], ("-1".to_string(), "3e-2".to_string()));
    }

    // =======================================================================
    // GPU plumbing (probe-gated): cupy candidates refined AND certified in
    // Rust; corrupted candidates either recovered from or refused — never
    // certified as-is.
    // =======================================================================

    #[test]
    fn gpu_cupy_plumbing_end_to_end() {
        if !cupy_available() {
            eprintln!("[gpu] cupy not importable — skipping (plumbing is optional by design)");
            return;
        }
        let n = 60usize;
        let prec: u32 = 212;
        let nlimbs = 4usize; // 212 = 4·53: lossless split
        let k = 2usize;
        let mut sigmas = vec![0.0; n];
        for (t, s) in sigmas.iter_mut().enumerate().skip(2) {
            *s = 0.4 + 1.6 * ((t - 2) as f64) / ((n - 3) as f64);
        }
        let (m, planted_kernel) = planted(n, &sigmas, &[0, 1], prec, 0x5eed_0003);
        let path = tmp_path("kernel_refine_gpu");
        write_ext(&m, nlimbs, &path);

        let cands = coarse_kernel_via_cupy(&path, k, n).expect("cupy helper must run on this box");
        assert_eq!(cands.len(), k);
        let mut opt = KernelRefineOptions::new(prec, "1e-40");
        opt.coarse = CoarseStage::External { candidates: cands };
        let ck = refine_kernel_streamed(&path, k, &opt).expect("GPU candidates must refine to certification");
        for (j, b) in ck.residual_bounds.iter().enumerate() {
            eprintln!("[gpu] certified residual {j}: {:.3e}", b.to_f64());
        }
        let mut worst = 0f64;
        for v in &ck.vectors {
            worst = worst.max(projection_residual(&planted_kernel, v, prec));
        }
        eprintln!("[gpu] worst projection residual onto planted kernel: {worst:.3e}");
        assert!(worst < 1e-20, "GPU-seeded refinement found the wrong subspace: {worst:.3e}");
        assert!(ck.provenance.contains("EXTERNAL"), "provenance must record the untrusted source");

        // deliberately corrupted candidates: identical garbage unit vectors.
        // Acceptable outcomes: recover (certified AND correct subspace) or
        // refuse (Err). NEVER a certificate over a wrong subspace.
        let garbage: Vec<Vec<(String, String)>> = (0..k)
            .map(|_| {
                let mut v = vec![("0".to_string(), "0".to_string()); n];
                v[0] = ("1".to_string(), "0".to_string());
                v
            })
            .collect();
        let mut opt_bad = KernelRefineOptions::new(prec, "1e-40");
        opt_bad.coarse = CoarseStage::External { candidates: garbage };
        match refine_kernel_streamed(&path, k, &opt_bad) {
            Ok(ck_bad) => {
                let target = Float::with_val(prec, Float::parse("1e-40").unwrap());
                for b in &ck_bad.residual_bounds {
                    assert!(*b <= target, "claimed success with a certificate above target");
                }
                let mut w = 0f64;
                for v in &ck_bad.vectors {
                    w = w.max(projection_residual(&planted_kernel, v, prec));
                }
                eprintln!("[gpu] corrupted candidates RECOVERED (projection residual {w:.3e})");
                assert!(w < 1e-20, "corrupted candidates 'certified' onto a wrong subspace: {w:.3e}");
            }
            Err(e) => eprintln!("[gpu] corrupted candidates honestly refused: {e}"),
        }
        std::fs::remove_file(&path).expect("cleanup");
    }

    // =======================================================================
    // LARGE probe (ignored): dim-1500 synthetic at reduced precision. Prints
    // measured wall times; the dim-3001/400-bit numbers in the report are
    // EXTRAPOLATIONS from these measurements, labeled as such.
    // =======================================================================

    #[test]
    #[ignore]
    fn large_probe_dim1500() {
        let n = 1500usize;
        let prec: u32 = 160;
        let k = 3usize;
        let mut sigmas = vec![0.0; n];
        for (t, s) in sigmas.iter_mut().enumerate().skip(3) {
            *s = 0.4 + 1.6 * ((t - 3) as f64) / ((n - 4) as f64);
        }
        let t0 = std::time::Instant::now();
        let (m, planted_kernel) = planted(n, &sigmas, &[0, 1, 2], prec, 0x5eed_0004);
        eprintln!("[probe1500] construction: {:.1?}", t0.elapsed());

        let t1 = std::time::Instant::now();
        let mc: Vec<Complex64> = coarse_copy(&m, prec);
        let lu = lu_factor(mc, n);
        eprintln!("[probe1500] f64 copy + LU ({n}³/3 flops): {:.1?}", t1.elapsed());
        drop(lu);

        let t2 = std::time::Instant::now();
        let mdd: Vec<DdC> = coarse_copy(&m, prec);
        let ludd = lu_factor(mdd, n);
        eprintln!("[probe1500] dd copy + LU: {:.1?}", t2.elapsed());
        drop(ludd);

        let v: Vec<MpC> = planted_kernel[0].clone();
        let t3 = std::time::Instant::now();
        let (_s, _t) = matvec_with_abs(&m, &v, prec);
        eprintln!("[probe1500] one certified hp matvec (n², {prec}-bit): {:.1?}", t3.elapsed());

        let t4 = std::time::Instant::now();
        let mut opt = KernelRefineOptions::new(prec, "1e-30");
        opt.with_gap = false; // the gap Gram/Cholesky is O(n³) hp — measured at
                              // dim 500 in planted_kernel_dim500_certified and
                              // EXTRAPOLATED (labeled) for larger dims.
        let ck = refine_kernel(&m, k, &opt).expect("probe refinement");
        eprintln!(
            "[probe1500] refine_kernel (k = {k}, target 1e-30): {:.1?} in {} iterations",
            t4.elapsed(),
            ck.iterations
        );
        let mut worst = 0f64;
        for w in &ck.vectors {
            worst = worst.max(projection_residual(&planted_kernel, w, prec));
        }
        eprintln!("[probe1500] projection residual onto planted kernel: {worst:.3e}");
        assert!(worst < 1e-10);
        eprintln!("[probe1500] certified residuals: {:?}", ck.residual_bounds.iter().map(|b| b.to_f64()).collect::<Vec<_>>());
        eprintln!("[probe1500] EXTRAPOLATION (not a measurement): dim-3001/400-bit scales the n² matvec by (3001/1500)² ≈ 4.0 and the 160→400-bit MPFR ops by ~2-3×; the dd LU scales by (3001/1500)³ ≈ 8.");
    }
}
