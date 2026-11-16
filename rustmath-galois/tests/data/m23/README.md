# Test data for the Steiner labelling of M23 polynomials

Used by `rustmath-galois/tests/steiner_labelling.rs`. The brief is
`docs/workstreams/m23_golay/HANDOFF.md`, Section 6 (last paragraph) and Section 8, step 2.

## Files

| File | Source | Licence | sha256 |
|---|---|---|---|
| `golay_resolvents_no_f.json` | derived from `johnjanik/M23`, `data/golay_resolvents.json`, commit `d05475b` (original sha256 `dcf1e38c4f13fc0b1c5503a1e2a074c6cdce2f74ea4dd292b0dc0855ee870d13`), by removing the `"f"` field of each entry (command below) | the owner's computed data, vendored as the brief allows | `87123dfe2fe86cb6bf5b40e2a39c2d06a680c6a2df2e8fd8bce1adee4f87a381` (133,201 bytes) |
| `frobenius_signs.json` | `johnjanik/M23`, `data/frobenius_signs.json`, commit `d05475b`, unchanged | the owner's data | `30ec467012f53e2b8e7e6a4d05e321a8adabe747aea1768488b9cda604617a4f` |
| `lelay_specialization_t2_Z.json` | Le Lay, `research-reflexivity/frontier-math-m23-cover-investigation`, `data/specialization_t2_Z.json`, commit `3f6ab4b`, unchanged | CC0-1.0 (`data/LICENSE` of that repository; its paper and code are CC BY-NC-SA and are not used) | `f775aa4c004541f25d121fac999fca9c62e493100f99982cc8253fbfa4445862` (line 84 of its `CHECKSUMS.sha256`) |
| `cc0_t2_steiner_reference.json` | new: reference values for the monic associate of the previous file, computed here (below) | derived from CC0 data | `91532572595a9cd6b3cc87473fca1dc416b32162c91c724bc91a507277f58dee` |
| `make_cc0_reference.py` | the script that wrote `cc0_t2_steiner_reference.json` | new | |

Le Lay's file is attributed to Knabble, Inc. (dba Reflexivity), who waived their rights under
CC0 1.0. It is the specialization at `T = 2` of the CC0 model `optimal_23_4_Z.json` that WP-F
vendors in `rustmath-curves/tests/data/m23/`. Its certificate (in the file) says that the
specialization is a subgroup of M23 that contains elements of order 23 (irreducible mod 29) and
14 (factor degrees 2, 7, 14 mod 11), hence equal to M23 (the only maximal subgroup of M23 of order
divisible by 23 is `23:11`, of odd order). The fast test `cc0_certificate_facts` checks the two
modular facts.

Not here: HJLPPZ's `polynomial_f1.gp` and `polynomial_f2.gp` (`shaowuz/m23isgalois`). That
repository has no licence, so the `env_` tests read them at run time from the directory named by
the environment variable `M23ISGALOIS`. They never appear in RustMath sources, comments or data.

## The derived file `golay_resolvents_no_f.json`

`data/golay_resolvents.json` of the M23 repository contains f1 and f2 themselves in its `"f"`
fields. They were removed with (Python 3.13.18):

```bash
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [v.pop("f") for v in d.values()]; json.dump(d, open(sys.argv[2],"w"), indent=0)' \
  data/golay_resolvents.json golay_resolvents_no_f.json
```

The fields kept are `labelling_prime`, `sign_at_prime`, `T5`, `J5`, `J7`, `R253_heptad`,
`R253_duad` (ascending coefficients, as decimal strings) and `verified_at_second_prime`.

Note for the owner: the file still determines f1 and f2. The power sums `Q_k` of the duad sums
`alpha_i + alpha_j` give the power sums of the roots by
`Q_k = ((46 - 2^k) P_k + sum_{0<m<k} C(k,m) P_m P_{k-m}) / 2` with `46 - 2^k != 0`, so `R253_duad`
recovers f by Newton's identities (review 1 of the design checked this for f2). Vendoring the
resolvents rests on the brief's permission for the owner's computed data.

## The reference file `cc0_t2_steiner_reference.json`

Values for the monic associate `g(x) = a^22 f(x/a)` of Le Lay's polynomial `f` (`a` its leading
coefficient; coefficients of `g` up to 1605 digits): the Fujiwara `best`, the bounds `B5`,
`B_J5`, `B_J7`, the labellings at 29 (`N = 270` and the minimal rigorous `N = 250`) and at 67
(`N = 216`) with both `T5` residues, `T5`, `J5`, `J7`, and the Frobenius signs at the twelve
inert primes below 900. Written by

```bash
python3 -m venv venv; venv/bin/pip install python-flint==0.9.0
venv/bin/python -I make_cc0_reference.py <frontier-math checkout>/data/specialization_t2_Z.json \
  cc0_t2_steiner_reference.json <checkout of johnjanik/M23 at d05475b>
```

which runs the owner's `m23golay/arith.py` (FLINT, an independent implementation). The output is
reproducible (same sha256 on reruns). The PARI/GP references of the design (`polrootspadic`) and
of its review (Newton in `Mod(Mod(., p^N), f)`) give the same values (field `second_reference`).
