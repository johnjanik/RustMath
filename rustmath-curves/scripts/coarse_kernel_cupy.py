#!/usr/bin/env python3
"""Optional GPU coarse stage for RustMath kernel refinement (E1a).

Reads a complex matrix dump, runs an FP64 SVD on the GPU (cupy), and writes the
k right-singular vectors of smallest singular value as decimal strings.

THE OUTPUT IS UNTRUSTED BY CONTRACT.  The Rust side (belyi::kernel_refine,
CoarseStage::External) uses these vectors only as starting candidates for
high-precision inverse-iteration refinement; every claim about the kernel is
re-derived there with rigorous MPFR certificates that are independent of this
script.  Garbage output costs iterations, never correctness.

Input formats (distinguished by exact file size):
  EXT limb dump (dump_scaled_ami_streamed / read_ext_matrix format):
      u32 dim, u8 nlimbs, then dim*dim entries row-major, each entry =
      nlimbs f64 Re limbs then nlimbs f64 Im limbs, little-endian.
      Size = 5 + dim*dim*2*nlimbs*8.  Limbs are summed in f64 (the coarse
      projection of the dumped value).
  RAW f64 dump (dump_2_12_5_matrix format):
      u32 dim, then dim*dim*(re,im) f64 pairs row-major, little-endian.
      Size = 4 + dim*dim*16.

Output (text, to OUT_PATH):
  line 1:  "<dim> <k>"
  then k blocks of dim lines, each "<re> <im>" in %.17g (round-trippable f64).

Usage:  coarse_kernel_cupy.py MATRIX_PATH K OUT_PATH
Exit nonzero (with a message on stderr) on any failure; writes nothing partial.
"""

import os
import struct
import sys


def read_matrix(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(5)
        if len(head) < 5:
            raise SystemExit(f"{path}: too short for any known dump format")
        (dim,) = struct.unpack("<I", head[:4])
        nlimbs = head[4]
        if dim == 0:
            raise SystemExit(f"{path}: dim = 0")
        if nlimbs > 0 and size == 5 + dim * dim * 2 * nlimbs * 8:
            fmt = ("ext", nlimbs)
        elif size == 4 + dim * dim * 16:
            fmt = ("raw", 1)
            f.seek(4)
        else:
            raise SystemExit(
                f"{path}: {size} bytes matches neither EXT (dim={dim}, nlimbs={nlimbs}) "
                f"nor RAW (dim={dim}) format"
            )
        import numpy as np

        if fmt[0] == "ext":
            data = np.fromfile(f, dtype="<f8", count=dim * dim * 2 * nlimbs)
            data = data.reshape(dim * dim * 2, nlimbs).sum(axis=1)
        else:
            data = np.fromfile(f, dtype="<f8", count=dim * dim * 2)
        m = data[0::2] + 1j * data[1::2]
        return m.reshape(dim, dim), fmt[0]


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    matrix_path, k_str, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(k_str)
    if k <= 0:
        raise SystemExit("K must be positive")
    m, fmt = read_matrix(matrix_path)
    dim = m.shape[0]
    if k >= dim:
        raise SystemExit(f"K = {k} must be < dim = {dim}")

    import cupy as cp

    mg = cp.asarray(m)
    # full_matrices=False is enough: we need right singular vectors (rows of vh)
    _, s, vh = cp.linalg.svd(mg, full_matrices=False)
    # singular values come back descending: the k smallest are the last k rows
    idx = list(range(dim - 1, dim - 1 - k, -1))
    vecs = cp.asnumpy(vh[idx, :]).conj()  # rows of Vh are conj(right vectors)
    sk = cp.asnumpy(s)
    print(
        f"[cupy] {fmt} dump dim={dim}: sigma_min={sk[-1]:.3e} "
        f"sigma_{{k+1}}={sk[dim - 1 - k]:.3e} (FP64, untrusted)",
        file=sys.stderr,
    )

    tmp = out_path + ".part"
    with open(tmp, "w") as f:
        f.write(f"{dim} {k}\n")
        for v in vecs:
            for z in v:
                f.write(f"{z.real:.17g} {z.imag:.17g}\n")
    os.replace(tmp, out_path)


if __name__ == "__main__":
    main()
