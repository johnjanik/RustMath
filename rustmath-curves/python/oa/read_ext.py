"""Reader for the extended-precision matrix dump (u32 dim, u8 nlimbs, row-major
re-limbs then im-limbs). Returns per-limb f64 arrays; the sum of limbs is the
dd/td value. Validates that sub-f64 limbs carry real information."""
import numpy as np, sys

def read_ext(path):
    with open(path, 'rb') as f:
        dim = int(np.frombuffer(f.read(4), dtype=np.uint32)[0])
        nlimbs = int(f.read(1)[0])
        raw = np.frombuffer(f.read(), dtype=np.float64)
    per = 2 * nlimbs
    assert raw.size == dim * dim * per, (raw.size, dim, per)
    raw = raw.reshape(dim, dim, 2, nlimbs)
    re_limbs = [raw[:, :, 0, l] for l in range(nlimbs)]
    im_limbs = [raw[:, :, 1, l] for l in range(nlimbs)]
    return dim, nlimbs, re_limbs, im_limbs

if __name__ == "__main__":
    path = sys.argv[1]
    dim, nlimbs, re, im = read_ext(path)
    print(f"dim={dim} nlimbs={nlimbs}")
    hi = re[0] + 1j * im[0]
    # relative magnitude of each successive limb vs the hi limb
    for l in range(1, nlimbs):
        rl = re[l] + 1j * im[l]
        mask = np.abs(hi) > 0
        rel = np.abs(rl[mask]) / np.abs(hi[mask])
        print(f"  limb {l}: max |limb|/|hi| = {rel.max():.2e}, median = {np.median(rel):.2e} "
              f"(expect ~1e-16^{l} = {10.0**(-16*l):.0e})")
    # non-overlap check: each limb should be < 2^-52 of the previous in magnitude (roughly)
    frac_nonzero = np.mean(np.abs(re[1]) > 0)
    print(f"  fraction of entries with nonzero lo limb: {frac_nonzero:.3f} (should be ~1.0)")
