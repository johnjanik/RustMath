//! High-precision complex one-sided Jacobi SVD (A = U Σ V*), in rug/MPFR.
//!
//! One-sided Jacobi is slower than bidiagonal SVD but far easier to make reliable in
//! arbitrary precision, and it delivers high *relative* accuracy for small singular
//! values and singular vectors (Drmač–Veselić; LAPACK xGEJSV philosophy). This is
//! exactly what the §4 modular-forms computation needs: the small-σ right-singular
//! subspace of A−I is the space S_k(Γ) of modular forms, which Gauss–Jordan pivots
//! could not resolve (they floor at ~10⁻⁴ while the true small σ ~10⁻²⁹).
//!
//! Complex scalars are stored as two `rug::Float`s (`MpC`) rather than `rug::Complex`,
//! which keeps Jacobi rotations allocation-light and exposes real-valued norms
//! directly. Adapted from a design note; degenerate/clustered singular values are
//! reported (individual vectors are then not canonical — use the subspace).
//!
//! # Sweep schedules and parallelism (E1b)
//!
//! Two pair orderings are available via [`JacobiSvdOptions::schedule`]:
//!
//! * [`JacobiSchedule::RowCyclic`] (default): the historical serial ordering
//!   (p, q) = (0,1), (0,2), …, (n−2, n−1). Bitwise-identical to the pre-E1b
//!   implementation; never uses rayon.
//! * [`JacobiSchedule::Tournament`]: a round-robin tournament (circle method).
//!   Each sweep is n−1 rounds (n rounds for odd n, one bye per round); each round
//!   pairs the n columns into ⌊n/2⌋ *disjoint* pairs, and every unordered pair
//!   (p, q) is visited exactly once per sweep. Pairs within a round are processed
//!   with `rayon` (`par_iter`).
//!
//! **Determinism argument for the tournament schedule.** A Jacobi step for the
//! pair (p, q) reads and writes only columns p and q of W and V. Within a round
//! the pairs are disjoint, so the steps of a round touch pairwise-disjoint column
//! sets: no step reads anything another step of the same round writes. Each step
//! is therefore a pure function of the round-entry state, and applying the steps
//! of a round in *any* order (or concurrently) produces the same round-exit state
//! — the rotations commute because they act on disjoint coordinates. Every scalar
//! operation is a correctly-rounded MPFR operation at fixed precision, hence
//! deterministic, and the schedule itself is a pure function of n. The per-sweep
//! reduction (max off-diagonal, rotation count) is folded in fixed pair order
//! from an order-preserving `collect`. Consequently the entire result is
//! **bitwise-identical at any thread count**; this is asserted by the
//! `tournament_determinism_1_4_8_threads` test. Convergence of parallel
//! round-robin orderings is classical (Luk & Park, *A proof of convergence for
//! two parallel Jacobi SVD algorithms*, 1989).
//!
//! Thread control: [`JacobiSvdOptions::threads`] = `Some(k)` runs the sweep loop
//! in a private rayon pool of k threads; `None` uses rayon's global pool. The
//! option is ignored by the serial `RowCyclic` schedule. Thread count never
//! affects results, only wall time.
//!
//! # Instrumentation and checkpointing (E2)
//!
//! Production runs of this kernel last hours to days; three such runs died with
//! zero forensic output. Two additions fix that, both **inert by default**:
//!
//! * [`JacobiSvdOptions::progress`]: an optional callback ([`SvdProgress`])
//!   receiving [`SvdEvent`]s — start, per-sweep (index, rotation count, max
//!   normalized off-diagonal, per-sweep and cumulative seconds), checkpoint
//!   written, done. [`SvdProgress::stderr`] emits one timestamped plain line per
//!   event, suitable for systemd-captured stdout/stderr. When `progress` is
//!   `None` no event is even constructed.
//! * [`JacobiSvdOptions::checkpoint_path`]: after each sweep (or every
//!   `checkpoint_every`-th sweep) the full iteration state is serialized to this
//!   path via write-to-temp + atomic `rename`, and [`jacobi_svd_resume`]
//!   continues from it. Because a sweep is a pure function of (W, V) and the
//!   schedule is fixed, a killed-and-resumed run is **bitwise-identical** to an
//!   uninterrupted one (asserted by `checkpoint_kill_and_resume_bitwise`). The
//!   checkpoint stores the last sweep's convergence predicate inputs (max
//!   off-diagonal, exact; changed flag) so the resumed control flow — including
//!   the reported sweep count — replays exactly.
//!
//! ## Checkpoint format v1 (line-oriented text)
//!
//! ```text
//! RUSTMATH-MPSVD-CKPT v1
//! prec <bits>
//! rows <r> cols <n>
//! schedule <row-cyclic|tournament>
//! sweep <k>
//! changed <0|1>
//! maxoff <hex float>            (radix-16, rug to_string_radix(16, None))
//! <r·n lines: W entries, column-major, "re_hex im_hex">
//! <n·n lines: V entries, column-major, "re_hex im_hex">
//! END fnv1a <16 hex digits>     (FNV-1a 64 over every preceding byte incl. newlines)
//! ```
//!
//! Radix-16 serialization round-trips `rug::Float` bit-exactly at the stored
//! precision (verified by `checkpoint_float_hex_roundtrip`). A truncated,
//! altered, or otherwise invalid file makes [`jacobi_svd_resume`] return an
//! honest [`SvdError::Checkpoint`] — never a silent restart from scratch.
//! Checkpointing requires all entries at the options' precision and is refused
//! otherwise (mixed precision cannot round-trip bit-exactly).

use rayon::prelude::*;
use rug::Float;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SvdError {
    DimensionMismatch,
    WideMatrixUnsupported,
    EmptyMatrix,
    InvalidTolerance,
    /// Checkpoint I/O, format, integrity, or compatibility failure.
    Checkpoint(String),
    /// The requested rayon thread pool could not be built.
    ThreadPool(String),
}

/// A high-precision complex number stored as (re, im).
#[derive(Clone, Debug)]
pub struct MpC {
    pub re: Float,
    pub im: Float,
}

impl MpC {
    pub fn zero(prec: u32) -> Self {
        Self { re: Float::with_val(prec, 0), im: Float::with_val(prec, 0) }
    }
    pub fn one(prec: u32) -> Self {
        Self { re: Float::with_val(prec, 1), im: Float::with_val(prec, 0) }
    }
    pub fn new(re: Float, im: Float) -> Self {
        Self { re, im }
    }
    pub fn from_f64(prec: u32, re: f64, im: f64) -> Self {
        Self { re: Float::with_val(prec, re), im: Float::with_val(prec, im) }
    }
    pub fn conj(&self) -> Self {
        Self { re: self.re.clone(), im: -self.im.clone() }
    }
    pub fn add(&self, rhs: &Self) -> Self {
        Self { re: self.re.clone() + &rhs.re, im: self.im.clone() + &rhs.im }
    }
    pub fn sub(&self, rhs: &Self) -> Self {
        Self { re: self.re.clone() - &rhs.re, im: self.im.clone() - &rhs.im }
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        let re = self.re.clone() * &rhs.re - self.im.clone() * &rhs.im;
        let im = self.re.clone() * &rhs.im + self.im.clone() * &rhs.re;
        Self { re, im }
    }
    /// conj(self) * rhs.
    pub fn conj_mul(&self, rhs: &Self) -> Self {
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.re.clone() * &rhs.im - self.im.clone() * &rhs.re;
        Self { re, im }
    }
    pub fn scale(&self, a: &Float) -> Self {
        Self { re: self.re.clone() * a, im: self.im.clone() * a }
    }
    pub fn div_real(&self, a: &Float) -> Self {
        Self { re: self.re.clone() / a, im: self.im.clone() / a }
    }
    pub fn abs2(&self) -> Float {
        self.re.clone() * &self.re + self.im.clone() * &self.im
    }
    pub fn abs(&self, prec: u32) -> Float {
        let mut x = Float::with_val(prec, self.abs2());
        x.sqrt_mut();
        x
    }
}

/// A dense complex matrix, row-major, at fixed precision.
#[derive(Clone, Debug)]
pub struct MpMatrix {
    pub rows: usize,
    pub cols: usize,
    pub prec: u32,
    pub data: Vec<MpC>,
}

impl MpMatrix {
    pub fn zeros(rows: usize, cols: usize, prec: u32) -> Self {
        Self { rows, cols, prec, data: vec![MpC::zero(prec); rows * cols] }
    }
    pub fn identity(n: usize, prec: u32) -> Self {
        let mut m = Self::zeros(n, n, prec);
        for i in 0..n {
            m.set(i, i, MpC::one(prec));
        }
        m
    }
    pub fn from_row_major(rows: usize, cols: usize, prec: u32, data: Vec<MpC>) -> Result<Self, SvdError> {
        if rows == 0 || cols == 0 {
            return Err(SvdError::EmptyMatrix);
        }
        if data.len() != rows * cols {
            return Err(SvdError::DimensionMismatch);
        }
        Ok(Self { rows, cols, prec, data })
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &MpC {
        &self.data[i * self.cols + j]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, z: MpC) {
        self.data[i * self.cols + j] = z;
    }
    pub fn col_norm2(&self, j: usize) -> Float {
        let mut acc = Float::with_val(self.prec, 0);
        for i in 0..self.rows {
            acc += self.get(i, j).abs2();
        }
        acc
    }
    /// ⟨col p, col q⟩ = Σ_i conj(A_ip) A_iq.
    pub fn col_dot(&self, p: usize, q: usize) -> MpC {
        let mut acc = MpC::zero(self.prec);
        for i in 0..self.rows {
            acc = acc.add(&self.get(i, p).conj_mul(self.get(i, q)));
        }
        acc
    }
    /// Right-multiply columns p,q by the 2×2 unitary diag(e,1)·[[c,s],[−s,c]].
    pub fn rotate_cols_complex_jacobi(&mut self, p: usize, q: usize, e: &MpC, c: &Float, s: &Float) {
        for i in 0..self.rows {
            let x = self.get(i, p).clone();
            let y = self.get(i, q).clone();
            let ex = x.mul(e);
            let new_p = ex.scale(c).sub(&y.scale(s));
            let new_q = ex.scale(s).add(&y.scale(c));
            self.set(i, p, new_p);
            self.set(i, q, new_q);
        }
    }
    pub fn normalize_columns_to_unit(&self, sigmas: &[Float]) -> Self {
        let mut u = Self::zeros(self.rows, self.cols, self.prec);
        for j in 0..self.cols {
            if sigmas[j] == 0 {
                continue;
            }
            for i in 0..self.rows {
                u.set(i, j, self.get(i, j).div_real(&sigmas[j]));
            }
        }
        u
    }
    pub fn permute_columns(&self, perm: &[usize]) -> Self {
        let mut out = Self::zeros(self.rows, self.cols, self.prec);
        for (new_j, &old_j) in perm.iter().enumerate() {
            for i in 0..self.rows {
                out.set(i, new_j, self.get(i, old_j).clone());
            }
        }
        out
    }
    /// Frobenius norm ‖A − U Σ V*‖ (validation).
    pub fn residual_norm(&self, u: &MpMatrix, sigmas: &[Float], v: &MpMatrix) -> Float {
        let mut acc = Float::with_val(self.prec, 0);
        for i in 0..self.rows {
            for j in 0..self.cols {
                let mut approx = MpC::zero(self.prec);
                for k in 0..self.cols {
                    let uk = u.get(i, k).scale(&sigmas[k]);
                    approx = approx.add(&uk.mul(&v.get(j, k).conj()));
                }
                acc += self.get(i, j).sub(&approx).abs2();
            }
        }
        acc.sqrt_mut();
        acc
    }
}

/// Pair-ordering schedule for the Jacobi sweeps. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JacobiSchedule {
    /// Historical serial ordering (0,1), (0,2), …; bitwise-identical to pre-E1b. Default.
    RowCyclic,
    /// Round-robin tournament: parallel-safe, bitwise-deterministic at any thread count.
    Tournament,
}

fn schedule_name(s: JacobiSchedule) -> &'static str {
    match s {
        JacobiSchedule::RowCyclic => "row-cyclic",
        JacobiSchedule::Tournament => "tournament",
    }
}

fn parse_schedule(s: &str) -> Option<JacobiSchedule> {
    match s {
        "row-cyclic" => Some(JacobiSchedule::RowCyclic),
        "tournament" => Some(JacobiSchedule::Tournament),
        _ => None,
    }
}

/// Progress events emitted through [`SvdProgress`]. Purely observational:
/// handlers cannot influence the computation.
#[derive(Debug)]
pub enum SvdEvent<'a> {
    Start {
        rows: usize,
        cols: usize,
        prec: u32,
        schedule: JacobiSchedule,
        /// 0 for a fresh run; the checkpoint's sweep index when resuming.
        start_sweep: usize,
    },
    SweepDone {
        /// 1-based cumulative sweep index (continues across resume).
        sweep: usize,
        /// Rotations actually applied this sweep.
        rotations: usize,
        /// Max normalized off-diagonal |⟨p,q⟩|/√(‖p‖²‖q‖²) seen this sweep.
        max_off: &'a Float,
        /// Wall seconds spent in this sweep.
        sweep_s: f64,
        /// Wall seconds since this call (fresh or resumed) started.
        elapsed_s: f64,
    },
    CheckpointWritten {
        sweep: usize,
        path: &'a Path,
        elapsed_s: f64,
    },
    Done {
        sweeps: usize,
        converged: bool,
        elapsed_s: f64,
    },
}

/// Shareable progress callback. `SvdProgress::stderr()` gives timestamped
/// plain-text lines; `SvdProgress::new` wraps any `Fn(&SvdEvent)`.
#[derive(Clone)]
pub struct SvdProgress(Arc<dyn Fn(&SvdEvent<'_>) + Send + Sync>);

impl SvdProgress {
    pub fn new(f: impl Fn(&SvdEvent<'_>) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
    /// One `[mp_svd] <unix-time> …` line per event on stderr (systemd-journal friendly).
    pub fn stderr() -> Self {
        Self::new(|ev| {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            match ev {
                SvdEvent::Start { rows, cols, prec, schedule, start_sweep } => eprintln!(
                    "[mp_svd] {ts:.3} start rows={rows} cols={cols} prec={prec} schedule={} start_sweep={start_sweep}",
                    schedule_name(*schedule)
                ),
                SvdEvent::SweepDone { sweep, rotations, max_off, sweep_s, elapsed_s } => eprintln!(
                    "[mp_svd] {ts:.3} sweep={sweep} rotations={rotations} max_off={max_off:.3e} sweep_s={sweep_s:.1} elapsed_s={elapsed_s:.1}"
                ),
                SvdEvent::CheckpointWritten { sweep, path, elapsed_s } => eprintln!(
                    "[mp_svd] {ts:.3} checkpoint sweep={sweep} path={} elapsed_s={elapsed_s:.1}",
                    path.display()
                ),
                SvdEvent::Done { sweeps, converged, elapsed_s } => eprintln!(
                    "[mp_svd] {ts:.3} done sweeps={sweeps} converged={converged} elapsed_s={elapsed_s:.1}"
                ),
            }
        })
    }
    fn emit(&self, ev: &SvdEvent<'_>) {
        (self.0)(ev)
    }
}

impl fmt::Debug for SvdProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SvdProgress(..)")
    }
}

#[derive(Clone, Debug)]
pub struct JacobiSvdOptions {
    pub prec: u32,
    pub max_sweeps: usize,
    pub tol: Float,
    pub sort_descending: bool,
    pub cluster_tol: Float,
    /// Sweep pair ordering; `RowCyclic` (default) preserves pre-E1b behavior exactly.
    pub schedule: JacobiSchedule,
    /// Tournament schedule only: `Some(k)` = private rayon pool of k threads
    /// (`Some(0)` = rayon's default sizing), `None` = rayon global pool.
    /// Never affects results, only wall time. Ignored by `RowCyclic`.
    pub threads: Option<usize>,
    /// Optional progress callback; `None` (default) emits nothing.
    pub progress: Option<SvdProgress>,
    /// Optional sweep-level checkpoint file (atomic-rename update).
    pub checkpoint_path: Option<PathBuf>,
    /// Write the checkpoint every k-th sweep (default 1 = every sweep; 0 = never).
    pub checkpoint_every: usize,
}

impl JacobiSvdOptions {
    pub fn new(prec: u32, max_sweeps: usize, tol_decimal: &str, cluster_decimal: &str) -> Self {
        let tol = Float::with_val(prec, Float::parse(tol_decimal).expect("valid tol"));
        let cluster_tol = Float::with_val(prec, Float::parse(cluster_decimal).expect("valid cluster tol"));
        Self {
            prec,
            max_sweeps,
            tol,
            sort_descending: true,
            cluster_tol,
            schedule: JacobiSchedule::RowCyclic,
            threads: None,
            progress: None,
            checkpoint_path: None,
            checkpoint_every: 1,
        }
    }
    pub fn with_schedule(mut self, schedule: JacobiSchedule) -> Self {
        self.schedule = schedule;
        self
    }
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }
    pub fn with_progress(mut self, progress: SvdProgress) -> Self {
        self.progress = Some(progress);
        self
    }
    pub fn with_checkpoint(mut self, path: impl Into<PathBuf>) -> Self {
        self.checkpoint_path = Some(path.into());
        self
    }
    pub fn with_checkpoint_every(mut self, every: usize) -> Self {
        self.checkpoint_every = every;
        self
    }
}

#[derive(Clone, Debug)]
pub struct JacobiSvdResult {
    pub u: MpMatrix,
    pub sigma: Vec<Float>,
    pub v: MpMatrix,
    pub sweeps: usize,
    pub final_offdiag: Float,
    pub clusters: Vec<Vec<usize>>,
}

impl JacobiSvdResult {
    /// Indices of singular values ≤ threshold — the numerical kernel dimension.
    pub fn numerical_nullity_indices(&self, threshold: &Float) -> Vec<usize> {
        self.sigma
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (s <= threshold).then_some(i))
            .collect()
    }
    /// Right singular vectors (columns of V) with σ ≤ threshold: an hp kernel basis of A.
    pub fn right_nullspace_basis(&self, threshold: &Float) -> MpMatrix {
        let idx = self.numerical_nullity_indices(threshold);
        let mut out = MpMatrix::zeros(self.v.rows, idx.len(), self.v.prec);
        for (new_j, &old_j) in idx.iter().enumerate() {
            for i in 0..self.v.rows {
                out.set(i, new_j, self.v.get(i, old_j).clone());
            }
        }
        out
    }
}

/// Thin one-sided complex Jacobi SVD of `a` (requires rows ≥ cols).
///
/// With default options this is the historical serial row-cyclic algorithm,
/// bitwise-identical to the pre-E1b implementation. See the module docs for the
/// parallel tournament schedule, progress instrumentation, and checkpointing.
pub fn jacobi_svd(a: &MpMatrix, opt: &JacobiSvdOptions) -> Result<JacobiSvdResult, SvdError> {
    if a.rows == 0 || a.cols == 0 {
        return Err(SvdError::EmptyMatrix);
    }
    if a.rows < a.cols {
        return Err(SvdError::WideMatrixUnsupported);
    }
    if opt.tol <= 0 {
        return Err(SvdError::InvalidTolerance);
    }
    let w = matrix_to_cols(a);
    let v = identity_cols(a.cols, opt.prec);
    run_jacobi(w, v, a.rows, a.prec, opt, 0, None)
}

/// Resume a checkpointed [`jacobi_svd`] run from `path` and drive it to
/// completion under `opt`.
///
/// `opt.prec` and `opt.schedule` must match the checkpoint (refused otherwise);
/// the remaining options are taken from `opt` — pass the same options as the
/// original run to reproduce it bit-exactly. A truncated or corrupted file is a
/// hard [`SvdError::Checkpoint`], never a silent restart.
pub fn jacobi_svd_resume(path: &Path, opt: &JacobiSvdOptions) -> Result<JacobiSvdResult, SvdError> {
    if opt.tol <= 0 {
        return Err(SvdError::InvalidTolerance);
    }
    let ck = read_checkpoint(path)?;
    if ck.prec != opt.prec {
        return Err(SvdError::Checkpoint(format!(
            "{}: precision mismatch (checkpoint {}, options {})",
            path.display(),
            ck.prec,
            opt.prec
        )));
    }
    if ck.schedule != opt.schedule {
        return Err(SvdError::Checkpoint(format!(
            "{}: schedule mismatch (checkpoint {}, options {})",
            path.display(),
            schedule_name(ck.schedule),
            schedule_name(opt.schedule)
        )));
    }
    run_jacobi(ck.w, ck.v, ck.rows, ck.prec, opt, ck.sweep, Some((ck.max_off, ck.changed)))
}

// ---------------------------------------------------------------------------
// Core iteration (shared by fresh and resumed runs)
// ---------------------------------------------------------------------------

fn matrix_to_cols(a: &MpMatrix) -> Vec<Vec<MpC>> {
    (0..a.cols)
        .map(|j| (0..a.rows).map(|i| a.get(i, j).clone()).collect())
        .collect()
}

fn identity_cols(n: usize, prec: u32) -> Vec<Vec<MpC>> {
    let mut v = vec![vec![MpC::zero(prec); n]; n];
    for (j, col) in v.iter_mut().enumerate() {
        col[j] = MpC::one(prec);
    }
    v
}

fn cols_to_matrix(cols: &[Vec<MpC>], rows: usize, prec: u32) -> MpMatrix {
    let n = cols.len();
    let mut m = MpMatrix::zeros(rows, n, prec);
    for (j, col) in cols.iter().enumerate() {
        for (i, z) in col.iter().enumerate() {
            m.set(i, j, z.clone());
        }
    }
    m
}

fn col_norm2_slice(col: &[MpC], prec: u32) -> Float {
    let mut acc = Float::with_val(prec, 0);
    for z in col {
        acc += z.abs2();
    }
    acc
}

fn col_dot_slice(p: &[MpC], q: &[MpC], prec: u32) -> MpC {
    let mut acc = MpC::zero(prec);
    for (x, y) in p.iter().zip(q.iter()) {
        acc = acc.add(&x.conj_mul(y));
    }
    acc
}

fn rotate_pair_slices(cp: &mut [MpC], cq: &mut [MpC], e: &MpC, c: &Float, s: &Float) {
    for i in 0..cp.len() {
        let x = cp[i].clone();
        let y = cq[i].clone();
        let ex = x.mul(e);
        cp[i] = ex.scale(c).sub(&y.scale(s));
        cq[i] = ex.scale(s).add(&y.scale(c));
    }
}

/// One Jacobi step on the column pair; arithmetic is operation-for-operation the
/// historical inner loop (`mat_prec` = the W entries' precision for Gram
/// accumulators, `prec` = the options' precision for the rotation scalars).
/// Returns (normalized off-diagonal if defined, whether a rotation was applied).
fn pair_step(
    wp: &mut [MpC],
    wq: &mut [MpC],
    vp: &mut [MpC],
    vq: &mut [MpC],
    tol: &Float,
    mat_prec: u32,
    prec: u32,
) -> (Option<Float>, bool) {
    let app = col_norm2_slice(wp, mat_prec);
    let aqq = col_norm2_slice(wq, mat_prec);
    if app == 0 || aqq == 0 {
        return (None, false);
    }
    let apq = col_dot_slice(wp, wq, mat_prec);
    let beta = apq.abs(prec);
    if beta == 0 {
        return (None, false);
    }
    let mut denom = Float::with_val(prec, app.clone() * &aqq);
    denom.sqrt_mut();
    let off = beta / denom;
    if off <= *tol {
        return (Some(off), false);
    }
    let (e, c, s) = jacobi_pair_rotation(&app, &aqq, &apq, prec);
    rotate_pair_slices(wp, wq, &e, &c, &s);
    rotate_pair_slices(vp, vq, &e, &c, &s);
    (Some(off), true)
}

fn two_cols_mut(cols: &mut [Vec<MpC>], p: usize, q: usize) -> (&mut Vec<MpC>, &mut Vec<MpC>) {
    debug_assert!(p < q);
    let (head, tail) = cols.split_at_mut(q);
    (&mut head[p], &mut tail[0])
}

fn row_cyclic_sweep(
    w: &mut [Vec<MpC>],
    v: &mut [Vec<MpC>],
    tol: &Float,
    mat_prec: u32,
    prec: u32,
) -> (Float, bool, usize) {
    let n = w.len();
    let mut changed = false;
    let mut rotations = 0usize;
    let mut max_off = Float::with_val(prec, 0);
    for p in 0..n {
        for q in (p + 1)..n {
            let (wp, wq) = two_cols_mut(w, p, q);
            let (vp, vq) = two_cols_mut(v, p, q);
            let (off, rotated) = pair_step(wp, wq, vp, vq, tol, mat_prec, prec);
            if let Some(off) = off {
                if off > max_off {
                    max_off = off;
                }
            }
            if rotated {
                changed = true;
                rotations += 1;
            }
        }
    }
    (max_off, changed, rotations)
}

/// Round-robin tournament rounds over 0..n (circle method). Every unordered
/// pair appears exactly once across the rounds; pairs within a round are
/// disjoint. n−1 rounds for even n, n rounds (one bye each) for odd n.
fn tournament_rounds(n: usize) -> Vec<Vec<(usize, usize)>> {
    if n < 2 {
        return Vec::new();
    }
    let m = if n % 2 == 0 { n } else { n + 1 }; // pad odd n with a bye slot m-1
    let mut players: Vec<usize> = (0..m).collect();
    let mut rounds = Vec::with_capacity(m - 1);
    for _ in 0..m - 1 {
        let mut pairs = Vec::with_capacity(m / 2);
        for i in 0..m / 2 {
            let a = players[i];
            let b = players[m - 1 - i];
            if a < n && b < n {
                pairs.push(if a < b { (a, b) } else { (b, a) });
            }
        }
        rounds.push(pairs);
        // Fix players[0]; rotate the rest one step.
        let last = players[m - 1];
        for i in (2..m).rev() {
            players[i] = players[i - 1];
        }
        players[1] = last;
    }
    rounds
}

struct PairTask {
    p: usize,
    q: usize,
    wp: Vec<MpC>,
    wq: Vec<MpC>,
    vp: Vec<MpC>,
    vq: Vec<MpC>,
}

fn tournament_sweep(
    w: &mut [Vec<MpC>],
    v: &mut [Vec<MpC>],
    rounds: &[Vec<(usize, usize)>],
    tol: &Float,
    mat_prec: u32,
    prec: u32,
) -> (Float, bool, usize) {
    let mut changed = false;
    let mut rotations = 0usize;
    let mut max_off = Float::with_val(prec, 0);
    for round in rounds {
        // Take the (disjoint) columns of this round out of w/v so each task
        // owns its data; `collect` preserves pair order, so the fold below is
        // a fixed-order reduction independent of thread count.
        let tasks: Vec<PairTask> = round
            .iter()
            .map(|&(p, q)| PairTask {
                p,
                q,
                wp: std::mem::take(&mut w[p]),
                wq: std::mem::take(&mut w[q]),
                vp: std::mem::take(&mut v[p]),
                vq: std::mem::take(&mut v[q]),
            })
            .collect();
        let done: Vec<(PairTask, Option<Float>, bool)> = tasks
            .into_par_iter()
            .map(|mut t| {
                let (off, rotated) = {
                    let PairTask { wp, wq, vp, vq, .. } = &mut t;
                    pair_step(wp, wq, vp, vq, tol, mat_prec, prec)
                };
                (t, off, rotated)
            })
            .collect();
        for (t, off, rotated) in done {
            w[t.p] = t.wp;
            w[t.q] = t.wq;
            v[t.p] = t.vp;
            v[t.q] = t.vq;
            if let Some(off) = off {
                if off > max_off {
                    max_off = off;
                }
            }
            if rotated {
                changed = true;
                rotations += 1;
            }
        }
    }
    (max_off, changed, rotations)
}

#[allow(clippy::too_many_arguments)]
fn run_sweeps(
    w: &mut Vec<Vec<MpC>>,
    v: &mut Vec<Vec<MpC>>,
    rows: usize,
    mat_prec: u32,
    opt: &JacobiSvdOptions,
    start_sweep: usize,
    resume_pred: Option<(Float, bool)>,
    t0: Instant,
) -> Result<(usize, bool), SvdError> {
    let prec = opt.prec;
    let n = w.len();
    // Replay the convergence check the interrupted run performed after the
    // checkpointed sweep, so control flow (and the sweep count) match an
    // uninterrupted run bit-for-bit.
    if let Some((max_off, changed)) = resume_pred {
        if !changed || max_off <= opt.tol {
            return Ok((start_sweep, true));
        }
    }
    let rounds = match opt.schedule {
        JacobiSchedule::Tournament => tournament_rounds(n),
        JacobiSchedule::RowCyclic => Vec::new(),
    };
    let mut sweeps_done = start_sweep;
    let mut converged = false;
    for sweep in start_sweep..opt.max_sweeps {
        let sweep_t = Instant::now();
        let (max_off, changed, rotations) = match opt.schedule {
            JacobiSchedule::RowCyclic => row_cyclic_sweep(w, v, &opt.tol, mat_prec, prec),
            JacobiSchedule::Tournament => tournament_sweep(w, v, &rounds, &opt.tol, mat_prec, prec),
        };
        sweeps_done = sweep + 1;
        if let Some(pr) = &opt.progress {
            pr.emit(&SvdEvent::SweepDone {
                sweep: sweeps_done,
                rotations,
                max_off: &max_off,
                sweep_s: sweep_t.elapsed().as_secs_f64(),
                elapsed_s: t0.elapsed().as_secs_f64(),
            });
        }
        if let Some(path) = &opt.checkpoint_path {
            if opt.checkpoint_every > 0 && sweeps_done % opt.checkpoint_every == 0 {
                write_checkpoint(path, w, v, rows, mat_prec, opt, sweeps_done, &max_off, changed)?;
                if let Some(pr) = &opt.progress {
                    pr.emit(&SvdEvent::CheckpointWritten {
                        sweep: sweeps_done,
                        path,
                        elapsed_s: t0.elapsed().as_secs_f64(),
                    });
                }
            }
        }
        if !changed || max_off <= opt.tol {
            converged = true;
            break;
        }
    }
    Ok((sweeps_done, converged))
}

fn run_jacobi(
    mut w: Vec<Vec<MpC>>,
    mut v: Vec<Vec<MpC>>,
    rows: usize,
    mat_prec: u32,
    opt: &JacobiSvdOptions,
    start_sweep: usize,
    resume_pred: Option<(Float, bool)>,
) -> Result<JacobiSvdResult, SvdError> {
    let prec = opt.prec;
    let n = w.len();
    let t0 = Instant::now();
    if let Some(pr) = &opt.progress {
        pr.emit(&SvdEvent::Start { rows, cols: n, prec, schedule: opt.schedule, start_sweep });
    }
    let (sweeps_done, converged) = match (opt.schedule, opt.threads) {
        (JacobiSchedule::Tournament, Some(k)) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .map_err(|e| SvdError::ThreadPool(e.to_string()))?;
            pool.install(|| run_sweeps(&mut w, &mut v, rows, mat_prec, opt, start_sweep, resume_pred, t0))?
        }
        _ => run_sweeps(&mut w, &mut v, rows, mat_prec, opt, start_sweep, resume_pred, t0)?,
    };

    let w_m = cols_to_matrix(&w, rows, mat_prec);
    let mut v_m = cols_to_matrix(&v, n, prec);
    let mut sigmas = Vec::with_capacity(n);
    for j in 0..n {
        let mut s = w_m.col_norm2(j);
        s.sqrt_mut();
        sigmas.push(s);
    }
    let mut u = w_m.normalize_columns_to_unit(&sigmas);

    if opt.sort_descending {
        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_by(|&i, &j| sigmas[j].partial_cmp(&sigmas[i]).unwrap());
        sigmas = perm.iter().map(|&i| sigmas[i].clone()).collect();
        u = u.permute_columns(&perm);
        v_m = v_m.permute_columns(&perm);
    }

    let final_offdiag = max_normalized_offdiag(&w_m);
    let clusters = detect_singular_value_clusters(&sigmas, &opt.cluster_tol, prec);
    if let Some(pr) = &opt.progress {
        pr.emit(&SvdEvent::Done { sweeps: sweeps_done, converged, elapsed_s: t0.elapsed().as_secs_f64() });
    }
    Ok(JacobiSvdResult { u, sigma: sigmas, v: v_m, sweeps: sweeps_done, final_offdiag, clusters })
}

/// Complex Jacobi rotation for the 2×2 Hermitian Gram block [[app, apq],[conj apq, aqq]]:
/// strip the phase of apq, diagonalize the real symmetric block, return diag(e,1)·[[c,s],[−s,c]].
fn jacobi_pair_rotation(app: &Float, aqq: &Float, apq: &MpC, prec: u32) -> (MpC, Float, Float) {
    let beta = apq.abs(prec);
    let e = if beta == 0 { MpC::one(prec) } else { apq.div_real(&beta) };
    let two = Float::with_val(prec, 2);
    let tau = (aqq.clone() - app) / (two * beta.clone());
    let abs_tau = tau.clone().abs();
    let mut root = Float::with_val(prec, tau.clone() * &tau + 1);
    root.sqrt_mut();
    let denom = abs_tau + root;
    let mut t = Float::with_val(prec, 1) / denom;
    if tau < 0 {
        t = -t;
    }
    let mut c = Float::with_val(prec, t.clone() * &t + 1);
    c.sqrt_mut();
    c = Float::with_val(prec, 1) / c;
    let s = c.clone() * t;
    (e, c, s)
}

fn max_normalized_offdiag(w: &MpMatrix) -> Float {
    let prec = w.prec;
    let n = w.cols;
    let mut max_off = Float::with_val(prec, 0);
    for p in 0..n {
        for q in (p + 1)..n {
            let app = w.col_norm2(p);
            let aqq = w.col_norm2(q);
            if app == 0 || aqq == 0 {
                continue;
            }
            let beta = w.col_dot(p, q).abs(prec);
            let mut denom = Float::with_val(prec, app * &aqq);
            denom.sqrt_mut();
            let off = beta / denom;
            if off > max_off {
                max_off = off;
            }
        }
    }
    max_off
}

fn detect_singular_value_clusters(sigmas: &[Float], cluster_tol: &Float, prec: u32) -> Vec<Vec<usize>> {
    if sigmas.is_empty() {
        return Vec::new();
    }
    let mut clusters = Vec::<Vec<usize>>::new();
    let mut current = vec![0usize];
    for i in 1..sigmas.len() {
        let (a, b) = (&sigmas[i - 1], &sigmas[i]);
        let gap = (a.clone() - b).abs();
        let scale = if a > b { a.clone() } else { b.clone() };
        let rel_gap = if scale == 0 { Float::with_val(prec, 0) } else { gap / scale };
        if rel_gap <= *cluster_tol {
            current.push(i);
        } else {
            if current.len() > 1 {
                clusters.push(current);
            }
            current = vec![i];
        }
    }
    if current.len() > 1 {
        clusters.push(current);
    }
    clusters
}

// ---------------------------------------------------------------------------
// Checkpoint I/O (format v1; see module docs)
// ---------------------------------------------------------------------------

const CKPT_MAGIC: &str = "RUSTMATH-MPSVD-CKPT v1";
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a64_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn ck_err(path: &Path, msg: impl fmt::Display) -> SvdError {
    SvdError::Checkpoint(format!("{}: {}", path.display(), msg))
}

struct CheckpointState {
    prec: u32,
    rows: usize,
    schedule: JacobiSchedule,
    sweep: usize,
    changed: bool,
    max_off: Float,
    w: Vec<Vec<MpC>>,
    v: Vec<Vec<MpC>>,
}

fn put_line(out: &mut BufWriter<fs::File>, hash: &mut u64, line: &str, path: &Path) -> Result<(), SvdError> {
    *hash = fnv1a64_update(*hash, line.as_bytes());
    *hash = fnv1a64_update(*hash, b"\n");
    out.write_all(line.as_bytes())
        .and_then(|_| out.write_all(b"\n"))
        .map_err(|e| ck_err(path, format!("write failed: {e}")))
}

#[allow(clippy::too_many_arguments)]
fn write_checkpoint(
    path: &Path,
    w: &[Vec<MpC>],
    v: &[Vec<MpC>],
    rows: usize,
    mat_prec: u32,
    opt: &JacobiSvdOptions,
    sweep: usize,
    max_off: &Float,
    changed: bool,
) -> Result<(), SvdError> {
    let n = w.len();
    // Bit-exact resume requires every entry to reparse at one precision.
    if mat_prec != opt.prec {
        return Err(ck_err(
            path,
            format!(
                "checkpointing requires matrix precision == options precision ({mat_prec} != {})",
                opt.prec
            ),
        ));
    }
    for col in w.iter().chain(v.iter()) {
        for z in col {
            if z.re.prec() != opt.prec || z.im.prec() != opt.prec {
                return Err(ck_err(
                    path,
                    format!(
                        "non-uniform entry precision ({}/{} vs {}); refusing to checkpoint",
                        z.re.prec(),
                        z.im.prec(),
                        opt.prec
                    ),
                ));
            }
        }
    }
    let tmp = PathBuf::from(format!("{}.tmp", path.display()));
    let file = fs::File::create(&tmp).map_err(|e| ck_err(&tmp, format!("create failed: {e}")))?;
    let mut out = BufWriter::new(file);
    let mut hash = FNV_OFFSET;
    put_line(&mut out, &mut hash, CKPT_MAGIC, path)?;
    put_line(&mut out, &mut hash, &format!("prec {}", opt.prec), path)?;
    put_line(&mut out, &mut hash, &format!("rows {rows} cols {n}"), path)?;
    put_line(&mut out, &mut hash, &format!("schedule {}", schedule_name(opt.schedule)), path)?;
    put_line(&mut out, &mut hash, &format!("sweep {sweep}"), path)?;
    put_line(&mut out, &mut hash, &format!("changed {}", u8::from(changed)), path)?;
    put_line(&mut out, &mut hash, &format!("maxoff {}", max_off.to_string_radix(16, None)), path)?;
    for col in w.iter().chain(v.iter()) {
        for z in col {
            let line = format!("{} {}", z.re.to_string_radix(16, None), z.im.to_string_radix(16, None));
            put_line(&mut out, &mut hash, &line, path)?;
        }
    }
    let end = format!("END fnv1a {hash:016x}");
    out.write_all(end.as_bytes())
        .and_then(|_| out.write_all(b"\n"))
        .map_err(|e| ck_err(path, format!("write failed: {e}")))?;
    let file = out
        .into_inner()
        .map_err(|e| ck_err(path, format!("flush failed: {e}")))?;
    file.sync_all().map_err(|e| ck_err(path, format!("fsync failed: {e}")))?;
    drop(file);
    fs::rename(&tmp, path).map_err(|e| ck_err(path, format!("atomic rename failed: {e}")))?;
    Ok(())
}

struct CkptReader<'a> {
    rdr: BufReader<fs::File>,
    hash: u64,
    path: &'a Path,
}

impl CkptReader<'_> {
    /// Next line, folded into the running checksum. Errors on EOF or a missing
    /// trailing newline (both mean truncation for hashed lines).
    fn line_hashed(&mut self) -> Result<String, SvdError> {
        let mut raw = String::new();
        let nread = self
            .rdr
            .read_line(&mut raw)
            .map_err(|e| ck_err(self.path, format!("read failed: {e}")))?;
        if nread == 0 {
            return Err(ck_err(self.path, "unexpected end of file (truncated checkpoint)"));
        }
        self.hash = fnv1a64_update(self.hash, raw.as_bytes());
        if !raw.ends_with('\n') {
            return Err(ck_err(self.path, "truncated checkpoint (line without newline)"));
        }
        raw.pop();
        Ok(raw)
    }
    /// Next line WITHOUT hashing (for the END line); `None` at EOF.
    fn line_raw(&mut self) -> Result<Option<String>, SvdError> {
        let mut raw = String::new();
        let nread = self
            .rdr
            .read_line(&mut raw)
            .map_err(|e| ck_err(self.path, format!("read failed: {e}")))?;
        if nread == 0 {
            return Ok(None);
        }
        if raw.ends_with('\n') {
            raw.pop();
        }
        Ok(Some(raw))
    }
}

fn parse_hex_float(tok: &str, prec: u32, path: &Path) -> Result<Float, SvdError> {
    Float::parse_radix(tok, 16)
        .map(|inc| Float::with_val(prec, inc))
        .map_err(|e| ck_err(path, format!("bad float '{tok}': {e}")))
}

fn header_field<'a>(line: &'a str, key: &str, path: &Path) -> Result<&'a str, SvdError> {
    line.strip_prefix(key)
        .and_then(|rest| rest.strip_prefix(' '))
        .ok_or_else(|| ck_err(path, format!("malformed header line '{line}' (expected '{key} …')")))
}

fn read_cols(r: &mut CkptReader<'_>, ncols: usize, len: usize, prec: u32) -> Result<Vec<Vec<MpC>>, SvdError> {
    let mut cols = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let mut col = Vec::with_capacity(len);
        for _ in 0..len {
            let line = r.line_hashed()?;
            let mut it = line.split_whitespace();
            let (re_tok, im_tok) = match (it.next(), it.next(), it.next()) {
                (Some(a), Some(b), None) => (a, b),
                _ => return Err(ck_err(r.path, format!("malformed entry line '{line}'"))),
            };
            let re = parse_hex_float(re_tok, prec, r.path)?;
            let im = parse_hex_float(im_tok, prec, r.path)?;
            col.push(MpC::new(re, im));
        }
        cols.push(col);
    }
    Ok(cols)
}

fn read_checkpoint(path: &Path) -> Result<CheckpointState, SvdError> {
    let file = fs::File::open(path).map_err(|e| ck_err(path, format!("open failed: {e}")))?;
    let mut r = CkptReader { rdr: BufReader::new(file), hash: FNV_OFFSET, path };

    let magic = r.line_hashed()?;
    if magic != CKPT_MAGIC {
        return Err(ck_err(path, format!("bad magic/version line '{magic}'")));
    }
    let prec: u32 = header_field(&r.line_hashed()?, "prec", path)?
        .parse()
        .map_err(|e| ck_err(path, format!("bad prec: {e}")))?;
    if prec < 2 {
        return Err(ck_err(path, format!("implausible precision {prec}")));
    }
    let dims_line = r.line_hashed()?;
    let dims: Vec<&str> = dims_line.split_whitespace().collect();
    let (rows, cols) = match dims.as_slice() {
        ["rows", r_, "cols", c_] => {
            let rows: usize = r_.parse().map_err(|e| ck_err(path, format!("bad rows: {e}")))?;
            let cols: usize = c_.parse().map_err(|e| ck_err(path, format!("bad cols: {e}")))?;
            (rows, cols)
        }
        _ => return Err(ck_err(path, format!("malformed dims line '{dims_line}'"))),
    };
    if rows == 0 || cols == 0 || rows < cols {
        return Err(ck_err(path, format!("invalid dimensions rows={rows} cols={cols}")));
    }
    // Refuse absurd headers before allocating (the checksum is only verifiable
    // at the end of the file).
    let entries = rows
        .checked_mul(cols)
        .and_then(|wc| cols.checked_mul(cols).map(|vc| (wc, vc)))
        .filter(|&(wc, vc)| wc.saturating_add(vc) <= 1_000_000_000)
        .ok_or_else(|| ck_err(path, format!("implausible dimensions rows={rows} cols={cols}")))?;
    let schedule_tok = header_field(&r.line_hashed()?, "schedule", path)?.to_string();
    let schedule = parse_schedule(&schedule_tok)
        .ok_or_else(|| ck_err(path, format!("unknown schedule '{schedule_tok}'")))?;
    let sweep: usize = header_field(&r.line_hashed()?, "sweep", path)?
        .parse()
        .map_err(|e| ck_err(path, format!("bad sweep: {e}")))?;
    let changed = match header_field(&r.line_hashed()?, "changed", path)? {
        "0" => false,
        "1" => true,
        other => return Err(ck_err(path, format!("bad changed flag '{other}'"))),
    };
    let maxoff_tok = header_field(&r.line_hashed()?, "maxoff", path)?.to_string();
    let max_off = parse_hex_float(&maxoff_tok, prec, path)?;

    let (_wc, _vc) = entries;
    let w = read_cols(&mut r, cols, rows, prec)?;
    let v = read_cols(&mut r, cols, cols, prec)?;

    let expect_end = format!("END fnv1a {:016x}", r.hash);
    let end_line = r
        .line_raw()?
        .ok_or_else(|| ck_err(path, "unexpected end of file (missing END line)"))?;
    if end_line != expect_end {
        return Err(ck_err(
            path,
            format!("integrity check failed (corrupted checkpoint): got '{end_line}'"),
        ));
    }
    if r.line_raw()?.is_some() {
        return Err(ck_err(path, "trailing data after END line"));
    }
    Ok(CheckpointState { prec, rows, schedule, sweep, changed, max_off, w, v })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PREC: u32 = 256;

    fn opts() -> JacobiSvdOptions {
        JacobiSvdOptions::new(PREC, 60, "1e-70", "1e-40")
    }

    // Diagonal matrix: singular values are the |diagonal|, sorted.
    #[test]
    fn svd_diagonal() {
        let data = vec![
            MpC::from_f64(PREC, 3.0, 0.0),
            MpC::zero(PREC),
            MpC::zero(PREC),
            MpC::from_f64(PREC, 0.0, -5.0),
        ];
        let a = MpMatrix::from_row_major(2, 2, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        assert!((r.sigma[0].to_f64() - 5.0).abs() < 1e-60);
        assert!((r.sigma[1].to_f64() - 3.0).abs() < 1e-60);
        assert!(a.residual_norm(&r.u, &r.sigma, &r.v).to_f64() < 1e-60);
    }

    // The note's 3×2 example: reconstruction residual must be ~0.
    #[test]
    fn svd_reconstructs() {
        let data = vec![
            MpC::from_f64(PREC, 1.0, 0.0),
            MpC::from_f64(PREC, 0.0, 1.0),
            MpC::from_f64(PREC, 2.0, -1.0),
            MpC::from_f64(PREC, 1.0, 0.0),
            MpC::from_f64(PREC, -1.0, 0.5),
            MpC::from_f64(PREC, 0.25, -0.75),
        ];
        let a = MpMatrix::from_row_major(3, 2, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        assert!(a.residual_norm(&r.u, &r.sigma, &r.v).to_f64() < 1e-60);
        assert!(r.sigma[0] >= r.sigma[1]); // sorted
    }

    // Rank-deficient matrix: exactly one tiny singular value ⇒ numerical nullity 1,
    // and the recovered kernel vector v satisfies A v ≈ 0.
    #[test]
    fn svd_nullspace() {
        // columns c0, c1 with c2 = c0 + 2 c1  ⇒ rank 2, nullity 1 (3×3).
        let c = |re: f64, im: f64| MpC::from_f64(PREC, re, im);
        // rows of [c0 | c1 | c0+2c1]
        let col0 = [c(1.0, 0.0), c(0.0, 1.0), c(2.0, 0.0)];
        let col1 = [c(1.0, 1.0), c(-1.0, 0.0), c(0.0, 1.0)];
        let mut data = Vec::new();
        for i in 0..3 {
            data.push(col0[i].clone());
            data.push(col1[i].clone());
            data.push(col0[i].add(&col1[i].scale(&Float::with_val(PREC, 2))));
        }
        let a = MpMatrix::from_row_major(3, 3, PREC, data).unwrap();
        let r = jacobi_svd(&a, &opts()).unwrap();
        let thr = Float::with_val(PREC, Float::parse("1e-50").unwrap());
        let ker = r.numerical_nullity_indices(&thr);
        assert_eq!(ker.len(), 1, "expected nullity 1, σ = {:?}", r.sigma.iter().map(|s| s.to_f64()).collect::<Vec<_>>());
        // A v ≈ 0 for the kernel vector
        let vb = r.right_nullspace_basis(&thr);
        let mut resid = Float::with_val(PREC, 0);
        for i in 0..3 {
            let mut acc = MpC::zero(PREC);
            for j in 0..3 {
                acc = acc.add(&a.get(i, j).mul(vb.get(j, 0)));
            }
            resid += acc.abs2();
        }
        assert!(resid.to_f64() < 1e-80, "‖A v‖² = {:.2e}", resid.to_f64());
    }

    // ---------------- E1b/E2 test support ----------------

    /// Deterministic pseudorandom test matrix (PCG-style LCG on the seed;
    /// entries are exactly-representable f64 in (−1, 1)).
    fn synth(rows: usize, cols: usize, prec: u32, seed: u64) -> MpMatrix {
        let mut s = seed;
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let mut f = [0f64; 2];
            for slot in f.iter_mut() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *slot = ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
            }
            data.push(MpC::from_f64(prec, f[0], f[1]));
        }
        MpMatrix::from_row_major(rows, cols, prec, data).unwrap()
    }

    fn hex(f: &Float) -> String {
        format!("p{} {}", f.prec(), f.to_string_radix(16, None))
    }

    /// Bitwise equality of two results: sweep count, clusters, σ, U, V, off-norm.
    fn assert_bitwise_eq(a: &JacobiSvdResult, b: &JacobiSvdResult, ctx: &str) {
        assert_eq!(a.sweeps, b.sweeps, "{ctx}: sweep counts differ");
        assert_eq!(a.clusters, b.clusters, "{ctx}: clusters differ");
        assert_eq!(hex(&a.final_offdiag), hex(&b.final_offdiag), "{ctx}: final_offdiag differs");
        assert_eq!(a.sigma.len(), b.sigma.len(), "{ctx}: σ lengths differ");
        for (i, (x, y)) in a.sigma.iter().zip(&b.sigma).enumerate() {
            assert_eq!(hex(x), hex(y), "{ctx}: σ[{i}] differs");
        }
        assert_eq!(a.u.data.len(), b.u.data.len(), "{ctx}: U sizes differ");
        for (i, (x, y)) in a.u.data.iter().zip(&b.u.data).enumerate() {
            assert_eq!(hex(&x.re), hex(&y.re), "{ctx}: U[{i}].re differs");
            assert_eq!(hex(&x.im), hex(&y.im), "{ctx}: U[{i}].im differs");
        }
        assert_eq!(a.v.data.len(), b.v.data.len(), "{ctx}: V sizes differ");
        for (i, (x, y)) in a.v.data.iter().zip(&b.v.data).enumerate() {
            assert_eq!(hex(&x.re), hex(&y.re), "{ctx}: V[{i}].re differs");
            assert_eq!(hex(&x.im), hex(&y.im), "{ctx}: V[{i}].im differs");
        }
    }

    /// Unique checkpoint path in the system temp dir, removed on drop.
    struct TempCkpt(PathBuf);
    impl TempCkpt {
        fn new(tag: &str) -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!("rustmath_mp_svd_ckpt_{tag}_{}", std::process::id()));
            let _ = fs::remove_file(&p);
            Self(p)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempCkpt {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
            let _ = fs::remove_file(PathBuf::from(format!("{}.tmp", self.0.display())));
        }
    }

    // ---------------- E1b: tournament schedule ----------------

    // The circle method must cover every unordered pair exactly once per sweep,
    // with disjoint pairs inside each round (the determinism precondition).
    #[test]
    fn tournament_rounds_cover_all_pairs_once() {
        for n in [2usize, 3, 6, 7, 12] {
            let rounds = tournament_rounds(n);
            let mut seen = std::collections::BTreeSet::new();
            for round in &rounds {
                let mut used = std::collections::BTreeSet::new();
                for &(p, q) in round {
                    assert!(p < q && q < n, "n={n}: bad pair ({p},{q})");
                    assert!(used.insert(p) && used.insert(q), "n={n}: round not disjoint");
                    assert!(seen.insert((p, q)), "n={n}: pair ({p},{q}) repeated");
                }
            }
            assert_eq!(seen.len(), n * (n - 1) / 2, "n={n}: not all pairs covered");
        }
    }

    // Tournament and row-cyclic reach the same spectrum (different rotation
    // orders, same fixed point) and both reconstruct A.
    #[test]
    fn tournament_matches_row_cyclic_invariants() {
        let a = synth(12, 9, PREC, 0x0BE1F1);
        let r_serial = jacobi_svd(&a, &opts()).unwrap();
        let r_tour = jacobi_svd(&a, &opts().with_schedule(JacobiSchedule::Tournament)).unwrap();
        assert!(a.residual_norm(&r_tour.u, &r_tour.sigma, &r_tour.v).to_f64() < 1e-60);
        for (x, y) in r_serial.sigma.iter().zip(&r_tour.sigma) {
            let diff = (x.clone() - y).abs().to_f64();
            let scale = x.to_f64().max(1e-300);
            assert!(diff / scale < 1e-60, "σ diverged between schedules: {diff:e}");
        }
    }

    // The E1b acceptance gate: same matrix, 1 vs 4 vs 8 threads, results
    // bit-identical (σ AND vectors AND sweep count).
    #[test]
    fn tournament_determinism_1_4_8_threads() {
        let a = synth(24, 16, PREC, 0xD37E12);
        let run = |k: usize| {
            jacobi_svd(&a, &opts().with_schedule(JacobiSchedule::Tournament).with_threads(k)).unwrap()
        };
        let r1 = run(1);
        assert!(r1.sweeps > 1, "matrix converged too fast to exercise the schedule");
        assert!(a.residual_norm(&r1.u, &r1.sigma, &r1.v).to_f64() < 1e-60);
        let r4 = run(4);
        let r8 = run(8);
        assert_bitwise_eq(&r1, &r4, "1 vs 4 threads");
        assert_bitwise_eq(&r1, &r8, "1 vs 8 threads");
        // None = rayon global pool: same bits again.
        let rg = jacobi_svd(&a, &opts().with_schedule(JacobiSchedule::Tournament)).unwrap();
        assert_bitwise_eq(&r1, &rg, "1 thread vs global pool");
    }

    // ---------------- E2: instrumentation ----------------

    #[test]
    fn progress_events_fire_in_order() {
        use std::sync::Mutex;
        let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log2 = Arc::clone(&log);
        let progress = SvdProgress::new(move |ev| {
            let tag = match ev {
                SvdEvent::Start { start_sweep, .. } => format!("start@{start_sweep}"),
                SvdEvent::SweepDone { sweep, .. } => format!("sweep{sweep}"),
                SvdEvent::CheckpointWritten { sweep, .. } => format!("ckpt{sweep}"),
                SvdEvent::Done { converged, .. } => format!("done:{converged}"),
            };
            log2.lock().unwrap().push(tag);
        });
        let a = synth(8, 6, PREC, 0x5EED);
        let r = jacobi_svd(&a, &opts().with_progress(progress)).unwrap();
        let log = log.lock().unwrap();
        assert_eq!(log[0], "start@0");
        assert_eq!(*log.last().unwrap(), "done:true");
        let sweep_events: Vec<_> = log.iter().filter(|t| t.starts_with("sweep")).collect();
        assert_eq!(sweep_events.len(), r.sweeps, "one SweepDone per sweep");
        assert_eq!(*sweep_events[0], "sweep1");
    }

    // ---------------- E2: checkpoint / restart ----------------

    // Serialized Floats must round-trip bit-exactly (the checkpoint's core
    // requirement). Radix-16 with digits=None is exact for binary floats.
    #[test]
    fn checkpoint_float_hex_roundtrip() {
        let vals = vec![
            Float::with_val(PREC, 0),
            Float::with_val(PREC, 1),
            Float::with_val(PREC, -1.5),
            Float::with_val(PREC, Float::parse("1e-70").unwrap()),
            Float::with_val(PREC, Float::parse("-9.87654321e300").unwrap()),
            Float::with_val(PREC, 2).sqrt(),
            -Float::with_val(PREC, 3).sqrt() / Float::with_val(PREC, 7),
        ];
        for f in &vals {
            let s = f.to_string_radix(16, None);
            let back = Float::with_val(PREC, Float::parse_radix(&s, 16).unwrap());
            assert_eq!(hex(f), hex(&back), "hex round-trip changed bits for {s}");
        }
    }

    // The E2 acceptance gate: run k sweeps with checkpointing, resume, and the
    // final result is bitwise-identical to an uninterrupted run — including the
    // sweep count (the checkpoint replays the convergence predicate).
    #[test]
    fn checkpoint_kill_and_resume_bitwise() {
        let a = synth(20, 14, PREC, 0xC4E7);
        let tour = || opts().with_schedule(JacobiSchedule::Tournament).with_threads(2);

        let full = jacobi_svd(&a, &tour()).unwrap();
        assert!(full.sweeps >= 4, "need ≥4 sweeps for a meaningful kill at 2 (got {})", full.sweeps);

        let ck = TempCkpt::new("resume");
        // "Kill" after 2 sweeps: the sweep budget stops the run right after the
        // sweep-2 checkpoint was written.
        let mut partial_opt = tour().with_checkpoint(ck.path());
        partial_opt.max_sweeps = 2;
        let _partial = jacobi_svd(&a, &partial_opt).unwrap();

        let resumed = jacobi_svd_resume(ck.path(), &tour().with_checkpoint(ck.path())).unwrap();
        assert_bitwise_eq(&full, &resumed, "uninterrupted vs killed-and-resumed");
    }

    // Row-cyclic checkpointing follows the same contract.
    #[test]
    fn checkpoint_resume_row_cyclic_bitwise() {
        let a = synth(16, 10, PREC, 0x50BC1C);
        let full = jacobi_svd(&a, &opts()).unwrap();
        assert!(full.sweeps >= 3, "need ≥3 sweeps (got {})", full.sweeps);
        let ck = TempCkpt::new("rowcyc");
        let mut partial_opt = opts().with_checkpoint(ck.path());
        partial_opt.max_sweeps = 1;
        let _ = jacobi_svd(&a, &partial_opt).unwrap();
        let resumed = jacobi_svd_resume(ck.path(), &opts()).unwrap();
        assert_bitwise_eq(&full, &resumed, "row-cyclic uninterrupted vs resumed");
    }

    // A resumed checkpoint of an already-converged run must reproduce the run's
    // terminal state without doing extra sweeps.
    #[test]
    fn checkpoint_of_converged_run_resumes_to_same_result() {
        let a = synth(10, 7, PREC, 0xF1A15);
        let ck = TempCkpt::new("conv");
        let full = jacobi_svd(&a, &opts().with_checkpoint(ck.path())).unwrap();
        let resumed = jacobi_svd_resume(ck.path(), &opts()).unwrap();
        assert_bitwise_eq(&full, &resumed, "converged-run resume");
    }

    // Corruption in any form is an honest Err, never a silent restart/garbage.
    #[test]
    fn checkpoint_corruption_is_refused() {
        let a = synth(12, 8, PREC, 0xBADF00D);
        let ck = TempCkpt::new("corrupt");
        let mut opt = opts().with_schedule(JacobiSchedule::Tournament).with_checkpoint(ck.path());
        opt.max_sweeps = 2;
        let _ = jacobi_svd(&a, &opt).unwrap();
        let good = fs::read_to_string(ck.path()).unwrap();
        let resume_opt = opts().with_schedule(JacobiSchedule::Tournament);

        // Sanity: the pristine file resumes fine.
        assert!(jacobi_svd_resume(ck.path(), &resume_opt).is_ok());

        // 1. Truncation.
        fs::write(ck.path(), &good.as_bytes()[..good.len() * 3 / 5]).unwrap();
        assert!(matches!(jacobi_svd_resume(ck.path(), &resume_opt), Err(SvdError::Checkpoint(_))), "truncated file must be refused");

        // 2. Single-character corruption in a payload line (checksum breaks).
        let mut lines: Vec<String> = good.lines().map(String::from).collect();
        let target = &mut lines[9]; // an entry line (payload starts at line 8)
        let pos = target.find(|c: char| c.is_ascii_hexdigit()).unwrap();
        let old = target.as_bytes()[pos];
        let new = if old == b'0' { b'1' } else { b'0' };
        target.replace_range(pos..=pos, std::str::from_utf8(&[new]).unwrap());
        fs::write(ck.path(), lines.join("\n") + "\n").unwrap();
        assert!(matches!(jacobi_svd_resume(ck.path(), &resume_opt), Err(SvdError::Checkpoint(_))), "bit-flipped file must be refused");

        // Restore the good file; wrong-options resumes must also be refused.
        fs::write(ck.path(), &good).unwrap();
        let wrong_prec = JacobiSvdOptions::new(128, 60, "1e-30", "1e-20").with_schedule(JacobiSchedule::Tournament);
        assert!(matches!(jacobi_svd_resume(ck.path(), &wrong_prec), Err(SvdError::Checkpoint(_))), "prec mismatch must be refused");
        assert!(matches!(jacobi_svd_resume(ck.path(), &opts()), Err(SvdError::Checkpoint(_))), "schedule mismatch must be refused");

        // Nonexistent path.
        let ghost = TempCkpt::new("ghost");
        assert!(matches!(jacobi_svd_resume(ghost.path(), &resume_opt), Err(SvdError::Checkpoint(_))));
    }

    // ---------------- E1b: manual speedup benchmark ----------------

    // Not part of the suite: run explicitly, under the mandated memory cap, with
    //   cargo test -p rustmath-curves --lib bench_tournament -- --ignored --nocapture
    // Measures one full sweep at dim 400×400 / 256-bit for 1/4/8/16 threads and
    // asserts the results are bit-identical across thread counts.
    #[test]
    #[ignore = "manual benchmark; run with --ignored --nocapture under the memory cap"]
    fn bench_tournament_speedup_dim400() {
        let a = synth(400, 400, PREC, 0xBEEFCAFE);
        let mut base: Option<JacobiSvdResult> = None;
        for &k in &[1usize, 4, 8, 16] {
            let mut opt = opts().with_schedule(JacobiSchedule::Tournament).with_threads(k);
            opt.max_sweeps = 1;
            let t = Instant::now();
            let r = jacobi_svd(&a, &opt).unwrap();
            let dt = t.elapsed().as_secs_f64();
            eprintln!("[bench] dim=400 prec=256 threads={k} one_sweep_s={dt:.2}");
            match &base {
                None => base = Some(r),
                Some(b) => {
                    for (x, y) in b.sigma.iter().zip(&r.sigma) {
                        assert_eq!(hex(x), hex(y), "spectrum differs at {k} threads");
                    }
                }
            }
        }
    }
}

