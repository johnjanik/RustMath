# WP-F records

The records of work package WP-F (the Frobenius table of the M23 curve X). The brief is
`../HANDOFF.md`, Section 3; its Status block says what was reproduced and what is open.

| File | What |
|---|---|
| `design/WPF.md` | The design, with the prototype results and the test plan |
| `design/WPF.review1.md` | The design review (decision D1 rejected: Decided only for the nodal certificate) |
| `impl/WPF.md` | The implementation report: changes, tests, regenerated tables, follow-ups F-1 to F-11 |
| `impl/WPF.review1.md` | The code review (approved, six minor items) |

The files are copied unchanged from the session that wrote them. Paths of the form
`scratchpad/m23/...` name that session's scratch directory, which is not kept. The PARI/GP scripts
and raw outputs there are not copied. Some of them hold the coefficients of HJLPPZ's
`polynomial_F.m`, which has no licence and must not enter this repository.
