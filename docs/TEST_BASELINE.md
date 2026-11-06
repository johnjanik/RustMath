# Test baseline (2026-11-05)

`cargo test --workspace --no-fail-fast` on the consolidated branch (belyi + integrate-lava-galois +
main, commit `9c7e23f` and later doc-only commits), rustc 1.97.0, fresh Linux container.

**Result: 13,586 passed, 60 failed, 334 ignored.** Every workspace crate and test target compiles.

## Failures by target

| Target | Failed | Cause (first look) |
|---|--:|---|
| `rustmath-groups` lib | 18 | `libgap_group`/`libgap_mixin` (pure-Rust GAP name-alikes), `semidirect_product`, `nilpotent_lie_group`, `generic::test_merge_elements` assertion failures |
| `rustmath-groups` doctests | 8 | `groups_catalog`, `generic`, `semimonomial_transformation*` examples |
| `rustmath-groups --test galois_narrowing` | 3 | **Environment, now fixed:** the gitignored atlas data was absent. With `build_cycletypes.py` and `build_transitive_24.g` (see `rustmath-groups/data/README.md`) all 4 tests pass, so the baseline is 57 failures once the data is built |
| `rustmath-lieconformal` doctests | 13 | Doc examples out of date with the API |
| `rustmath-trees` doctests | 5 | Doc examples |
| `rustmath-numbertheory` doctests | 4 | Doc examples |
| `rustmath-quivers` lib | 2 | `cluster_quiver::test_cartan_companion`, `test_type_d_quiver` |
| `rustmath-modules` doctests | 2 | Doc examples |
| `rustmath-algebras` doctests | 2 | `associated_graded`, `cellular_basis` examples |
| `rustmath-curves` lib | 1 | `belyi::modular_forms_hp::recover_forms_centered_z_a_matches_uncentered`: numerical divergence 3.36e-10 (tolerance) |
| `rustmath-stats`, `rustmath-misc` doctests | 1 each | Doc examples |

The crates touched by the lava-galois merge (`rustmath-galois`, `-finitefields`, `-padics`,
`-numberfields`, `-functionfields`, `-polynomials`, `-reals`, `-complex`, `-integers`, `-igp24`)
have no failures. Whether the `rustmath-groups` lib failures predate the merge has not been
checked (the merge only auto-merged `rustmath-groups/src/lib.rs`); `PROJECT_TRACKER.md` records
the groups test target as "partial test-target repairs (stopped mid-edit at freeze)".

## Reproduce

```bash
cargo test --workspace --no-fail-fast 2>&1 | tee test_all.log
grep -E '^test .* FAILED$' test_all.log
```

The first build compiles GMP/MPFR from source (`rug`), which takes several minutes. The full
run took about an hour in a 4-core container.
