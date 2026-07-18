# RustMath build handoff #2 — the SVD wall (measured), and three smaller items

> Follow-up to `RUSTMATH_BUILD_HANDOFF.md` (all items delivered 12 Jul). This addendum
> is grounded in **measured** production attempts from 16–18 Jul (M23 campaign session,
> notebook §101). Everything below was hit in anger, not projected from reading code.

## The measured facts

| What | Measurement | Where |
|---|---|---|
| exact `phi_in_u` at len=201 | **68 min** single-core | `M23/route_a/phi_dump_201.log` |
| exact `phi_in_u` at len=1501 | **>13h47m, killed unfinished** (RSS 31 MB — still in the preamble) | ps forensics, §101 |
| hp φ (driver-side `phi_hp`) at len=1501 | **31.5 s** | `plumbing2_n1500_hp.log` |
| hp φ at len=3001 / 400-bit out | **496 s** (internal 6468-bit guard) | `phi_time_3001.log` |
| hp φ vs exact, len=201 / 401 | max rel err **3.3e-75 / 6.6e-75** (bound 1e-40), zeros exact, support exact | G1 gate runs |
| `recover_forms` chain, dim-1501 / 256-bit, single-core | **>24 h, killed at RuntimeMaxSec** (rc=143) | `plumbing2_n1500_hp.log` |
| dim-1001 / 256-bit assembly+SVD peak RSS | **5.96 GB** | `g2a_solve_hp_n1000.log` |
| hot loop threading | **single-core** (99.9% CPU with `RAYON_NUM_THREADS=8` set) | ps observation, 3 runs |
| dim-3001 / 400-bit single-core projection | **≥16 days**, ~84 GB peak (from the ≥24h dim-1501 bound × (3001/1501)³ × (400/256)^1.6) | §101 prereg item 6 |

Consequence: the `[2,12,5]` production solve (N=3000/prec=400/digits=12) is **hard-gated**
on E1 below. Even the N=1500 tier is impractical single-core.

## E1 — Fast hp kernel extraction (the gate) `P0` — two designs, build (a), keep (b) as fallback

**The reframing:** the pipeline consumes the KERNEL (the forms) plus a gap certificate
(sigma_min separated from sigma_next), not the full SVD. Full two-sided hp Jacobi pays
O(n^3)*sweeps at 400 bits for discarded information.

### E1a — mixed-precision kernel refinement (primary; minutes-to-hours)
- **Design:** (1) coarse kernel direction at LOW precision — GPU: the RTX 5090 on this
  box already ran FP64 SVD of this matrix class in 0.5 s (cupy, CUDA 12.8 — see memory
  `igp24-gpu-svd-setup`); per the campaign's own "FP64 wall" history (small singular
  values cluster below FP64 at N≈3000 — the reason the chain went hp), the coarse stage
  likely needs **double-double (~106-bit)** via FMA error-free transforms
  (CAMPARY-style; a dd LU of dim-3001 is ~0.25 s even at GeForce's 1:64 FP64 rate), or
  qd (~212-bit) if dd doesn't separate; a 24-core CPU 128-bit coarse stage is the
  GPU-free variant. (2) refinement — inverse iteration: each step = one low-precision
  solve (ms) + one **400-bit residual matvec** (9M complex MPFR ops ≈ 2 s single-core,
  parallelizes perfectly over rows → ~0.1–0.2 s on 24 cores); ~15 digits/iteration →
  400 bits in ~30 iterations. (3) **self-certification:** the refined kernel vector is
  verified a posteriori by one threaded hp matvec — ||Mv|| at 400 bits is a certificate
  independent of how v was found; the GPU is checked, never trusted. (4) gap
  certificate: a few subspace-iteration steps at ~200-bit with the same threaded matvec.
- **Memory bonus:** refinement needs only matvecs — the matrix can stay in the streamed
  EXT form (A8 reader) instead of materializing ~84 GB.
- **Acceptance:** on the (5,3,3) golden and a dim-≈500 case, refined kernel matches the
  serial Jacobi kernel to working precision; the 400-bit residual certificate and the
  gap certificate both emitted; end-to-end dim-3001/400-bit kernel in ≤ 2 h.

### E1b — 24-core one-sided Jacobi (fallback; ~1 day)
- **What:** parallelize the existing hp Jacobi (`mp_svd.rs::jacobi_svd`) — one-sided
  Hestenes with a round-robin tournament ordering gives n/2 disjoint column pairs per
  round; rayon over the pairs. Fixed schedule ⇒ bitwise-deterministic at any thread
  count. The current hot loop never engages rayon (observed single-core across three
  multi-hour runs).
- **Spec target:** dim-3001 / 400-bit in **8–16 h at 16 threads** (preregistered in
  notebook §101 item 6; ≥16-day single-core bound measured). Memory ≤ 90 GB or blocked
  via the EXT reader.
- **Acceptance:** bit-compatible kernel with the serial path on the (5,3,3) golden tests
  and a dim-≈500 comparison; near-linear speedup to ≥8 threads on dim-1501.

*(External zero-Rust-edit option, unmeasured: dump via A8's EXT format and run MPLAPACK
Cgesvd with threaded MPBLAS — worth an afternoon probe only if E1a/E1b stall.)*

## E2 — Progress + checkpoint instrumentation in the solve chain `P1`
- **What:** (a) phase timestamps (assembly done → SVD start → per-sweep line with sweep
  index and off-diagonal norm); (b) sweep-level checkpoint/restart so a RuntimeMaxSec
  kill doesn't discard a day of sweeps. Three multi-hour runs died with zero forensic
  output; phase attribution had to be inferred from RSS.
- **Acceptance:** a killed run can resume and its log names the phase and sweep it died in.

## E3 — Upstream the hp φ preamble (or memoize exact φ) `P1`
- **What:** `hypergeometric::phi_in_u`'s exact-rational series (inverse + product +
  revert with linearly growing coefficient heights) is unusable at production N — see
  the table. The campaign's driver-side fix (`M23/route_a/belyi_shakedown/src/phi_hp.rs`)
  computes φ in hp floats via the reduction **u^a = t·R(t)^a** (one Newton order-doubling
  reversion of a half-length series, per-step residual certificates, internal guard
  precision ~2.2× output). Lift it into `rustmath-curves` proper (keeping the exact path
  for small-N cross-validation), or alternatively memoize exact φ series to disk keyed by
  (a,b,c,len). G1 cross-validation harness already exists (`phi-check-file` in the driver).
- **Acceptance:** `run_belyi_2_12_5` production entry uses the hp (or memoized) path;
  exact-vs-hp agreement gate ≤1e-40 at two lengths wired as a test.

## E4 — Move the m/e valuation refusal before the expensive phase `P2`
- **What:** the echelon-valuation check (which refuses a wrong m/e guess with the
  valuation list) sits *after* forms recovery — a wrong guess costs a full multi-hour
  run. Either allow a cheap valuation pre-probe at small N with the same chart, or
  document the refusal cost prominently in the entry point.
- **Acceptance:** a wrong m/e guess is detectable in minutes, not hours.

## Context for the builder
The φ side is **done and certified** campaign-side (G1 at two lengths, refusal parity
with the exact path); E3 is upstreaming, not invention. E1 is the only item standing
between the campaign and the production `[2,12,5]` run — whose preregistered purpose
(notebook §101, item 5) is now **pipeline certification against the 7-July cubic-field
theorem**, not a direct M23/ℚ shot: the passport provably has no ℚ-member, so the snap
is expected to land in ℚ[x]/(x³−x²+5x−3) (disc −460). A ℚ landing would be an error
signal against one of two computations — treat it accordingly.
