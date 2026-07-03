# GPU Ozaki + Ogita–Aishima refined SVD — for the KMSV [2,12,5] null space

High-precision SVD of the horribly-conditioned KMSV matrix `M`, computed on the GPU by
emulating double-double precision with FP64 GEMMs (Ozaki scheme) inside an Ogita–Aishima
iterative-refinement loop. The goal is the 9-dimensional near-**null** subspace of `M`
(the weight-4 modular forms on the (2,12,5) triangle group) to ~double-double accuracy.

## Why this exists

The [2,12,5] Belyi map (the M₂₄→M₂₃ portal, Nielsen class σ₀=2⁸1⁸, σ₁=12², σ∞=5⁴1⁴,
degree 24, genus 0) is extracted from these forms. At **FP64** precision the degree-24
solve **overfits** the ~1e-6 form noise and returns a *generic* rational map, not the
*ramified* Belyi map — its zeros don't pair into 8 doubles, its fiber over 1 doesn't
collapse onto 2 order-12 points (they scatter over ~3.8× the coordinate scale). The
ramified map needs forms to ~1e-10 (N≈2500) and, for clean order-12 points, ~1e-24
(N≈5900).

`N` and arithmetic precision are **locked**: a clean SVD gap needs `p_bits ≳ N/36.7`
(from ρ^{2N} < 2^{-p}, ρ≈0.990605). So FP64 dies at N≈1900 (`ρ^{-N}·ε` roundoff floor
merges the `ρ^N` truncation floor). Pushing N further needs the arithmetic to follow —
hence GPU FP64-emulated double-double.

## Pipeline

```
RustMath  modular_forms_hp::dump_2_12_5_matrix_ext   (extended-precision dump of M)
   M_N=2500 M_K=4 M_PREC=140 M_LIMBS=2  → m_ext_n2500_dd.bin   (dd; ~20 min, rayon)
        │  format: u32 dim, u8 nlimbs, row-major (re limbs, im limbs) f64 LE
        ▼
read_ext.py    load the dd/td limbs
        ▼
mxpsvd.py      Ogita–Aishima refined SVD → dd-accurate 9-dim null space
        ▼   (recover forms: un-scale bₙ=ρ⁻ⁿyₙ, ×(1−w)^k, echelonize — see scratchpad)
   ramified Belyi map Φ=P/Q  →  FactorizedRoots seed  →  refine_hp  →  conic
```

## Files

- **`efp.py`** — error-free FP64 transforms (Dekker `two_sum`/`two_prod`) + mpmath ground
  truth for validating the GEMM. `python3 efp.py` self-tests the primitives.
- **`ozaki.py`** — the engine: Ozaki-scheme error-free **double-double complex GEMM** on
  the GPU (cupy). Splits each operand into low-bit slices so slice-products accumulate
  exactly in FP64, sums the partial GEMMs in dd. `python3 ozaki.py` validates vs mpmath:
  **n_slices=6 → 2e-32**, n_slices=4 → 2e-23 (naive FP64 GEMM only reaches 1e-14).
- **`ddcx.py`** — double-double complex elementwise arithmetic (add/sub/mul/div, conj,
  conj-transpose) for the O(n²) per-pair work in the refinement.
- **`mxpsvd.py`** — Algorithms 3 (`MxpSVDStep`, complex, 3-branch cluster-safe),
  2 (`ClusterRR`, Rayleigh–Ritz on the σ≈0 cluster), 4 (driver), from Schwarz et al.,
  *Mixed-Precision SVD on GPUs via Ogita–Aishima Iterative Refinement* (NVIDIA).
- **`read_ext.py`** — reader for the extended-precision matrix dump.

## Status (2026-07-03)

Infrastructure **validated**: Ozaki GEMM (2e-32), dd-complex arithmetic, the dd/td dump
(lo limb ~4e-17 of hi). The refinement **step does not yet converge** — the synthetic
self-test (`python3 mxpsvd.py`) shows `omega` stuck at ~1.4e-6 instead of dropping toward
`16 n εhp σmax ≈ 1e-22`. Ordered debug plan is in the `mxpsvd.py` module docstring:
1. test `mxp_svd_step` on an easy O(1)-σ matrix first (isolates step from conditioning);
2. fix a transient NaN at iter 0 (likely `log2(0)` in `ozaki_split` on an all-zero row);
3. hand-check the branch-A coupling (α=t_ij+σ_j r_ij, β=t̄_ji+σ_i s_ij) and the U(I+F) sign;
4. confirm the loop actually mutates U,V.

Once the step converges: assemble N=2500 dd, refine, recover forms, and check the order-12
clustering sharpens from FP64's scatter (~3.8) toward ~0.1 (N=2500) / ~0.01 (N=5900, td).

## GPU setup

RTX 5090 (Blackwell sm_120) needs CUDA ≥ 12.8 pip nvrtc wheels; `pip install
"nvidia-cuda-nvrtc-cu12>=12.8"` etc., and do **not** set `CUDA_PATH`. All modules fall back
to numpy (CPU) if cupy is unavailable.
