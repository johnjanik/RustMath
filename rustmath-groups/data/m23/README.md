# The ATLAS character table of M23 (WP-P)

`m23_atlas_table.json` is embedded by `src/m23/atlas.rs` (`include_str!`). It is written by
`m23_atlas_table.g` from GAP's CTblLib, `CharacterTable("M23")` (the ATLAS of Finite Groups, 1985,
M23, p. 71). Run from the workspace root:

```bash
gap -q rustmath-groups/data/m23/m23_atlas_table.g < /dev/null > rustmath-groups/data/m23/m23_atlas_table.json
sha256sum rustmath-groups/data/m23/m23_atlas_table.json
```

With GAP 4.12.1 and CTblLib 1.3.7 the output is the committed file byte for byte: 3,471 bytes,
sha256 `f9503e809aef25c369cf2b40c816bf72e23876d8162343cb75166e8d0f5196eb`.

A value `[x, y, d]` is `x + y b_d` with the ATLAS irrationality `b_d = (-1 + sqrt(-d)) / 2`,
`d` in `{7, 11, 15, 23}`; `[x, 0, 1]` is a rational value. The script reconstructs every value in
GAP and stops with an error if the encoding fails. The classes are in GAP's column order, which is
the order of `M23Class::ALL`. The licence footing of the generated table is described in
`tests/data/m23/README.md`.
