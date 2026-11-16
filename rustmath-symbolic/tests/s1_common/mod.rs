//! Shared code of the S1 test files (`s1_regressions.rs`, `s1_singular_points.rs`,
//! `s1_fuzz_data.rs`).
//!
//! S1 is the series engine of rustmath-symbolic (`Expr::taylor` rewritten as truncated Laurent
//! series over exact constants) and the literal parser. Specification: MAGMA Handbook ch. 49,
//! "Power, Laurent and Puiseux Series", §49.1.3-49.1.4 (precision model) and §49.5 (transcendental
//! functions), digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`; the S1
//! design (frozen at revision 7rt4 with revision 8), §2.4, §2.5, §2.9, §2.12, §2.13, §4.1, and the
//! owner's decision of 2026-11-15 (two rules: an argument whose series is an exact constant c is
//! evaluated as f(c) with no expansion and no integrand; a non-constant argument at a branch point
//! is `NotLaurent`, never `DivisionByZero`; `DivisionByZero` only when the input divides by an
//! exact zero). Recurrences: Knuth, TAOCP vol. 2, 3rd ed., §4.7; Brent and Kung, "Fast algorithms
//! for manipulating formal power series", J. ACM 25(4), 1978, 581-595.
//!
//! # Scope of the owner's first rule (every row of the three files follows it)
//!
//! The owner's text is about the inverse functions ("for each inverse function and each of its
//! branch points"; "that keeps the inverse functions everywhere they are regular"). The tests take
//! the scope of review 9, B1 required change 1, with the frozen design (§2.4, §2.5):
//!
//! - Mandatory. An argument of arcsin, arccos, arctan, arcsinh, arccosh or arctanh whose series is
//!   an exact constant c gives `apply(f, c)` before any integrand is formed: the value f(c) (a
//!   rational, or a kernel when c is not rational), or `DivisionByZero` at a pole of the constant
//!   (arctanh(+-1), arctan(+-I)). This holds at a branch point and at a regular point alike, so
//!   `sqrt(arcsin(x - x))` is 0. The rules that the design already states for an exact constant
//!   hold too: the exponent rule and the base rule of `Pow` (`x^(x - x + 2)`, `(x - x)^r`, `0^r`,
//!   `(1/2)^(sign(x^2 + 1))` = 1/2), and log and inv of the exact 0 (`DivisionByZero`).
//! - Exact constants are the series the design's exact rules produce: `x^0`, a monomial divided
//!   by itself (`x/x` at 0, `(x-1)/(x-1)` at 1), sign and abs of a series with rational
//!   coefficients (`sign(x^2 + 1)`), a constant factor 0, exact polynomial arithmetic (`x - x`,
//!   `(x+1)^2 - x^2 - 2*x - 1`), a power with an exact-constant exponent, and the six functions
//!   above of an exact constant. `x/x` at a nonzero point is not exact (the inverse of a
//!   non-monomial is truncated, review 9 m3): `PrecisionExhausted` or the value.
//! - Optional. The same evaluation for exp, sin, cos, tan, sinh, cosh, tanh, erf, log of a nonzero
//!   constant, and abs and sign of an exact constant that is not rational (§2.4 keeps abs and sign
//!   of non-rational coefficients `Unsupported`). A port may fold these or not. Where a row's
//!   outcome depends on such a fold, the row accepts each honest outcome (`check::Outcome`, the
//!   ` | ` rows of the data files, class `O` of the fuzz data): `atanh(exp(x - x))` is
//!   `PrecisionExhausted` (exp of the exact 0 is `1 + O(h^P)`) or `DivisionByZero` (it is the exact
//!   1, a pole of arctanh); `sign(arcsin(sign((x+1/2)^2)))` at -1/2 is `Unsupported` or 1.
//!   The rows therefore pass under review 9's scope and under the general reading of the rule
//!   (every function of an exact constant evaluated), and fail only on a wrong value, a wrong
//!   refusal kind, a panic or a budget hole.
//!
//! The files use the public API that the design specifies (§2.9: `series_expansion`,
//! `SeriesExpansion`, `to_expr`, `taylor_with_budget`, `SeriesError`; §2.13: `limit_via_series`,
//! `limit_with_error`). They were written before the implementation, behind a feature gate that
//! the implementation removed. `api.rs` is the only file that names these entry points.
#![allow(dead_code)]

pub mod api;
pub mod check;
pub mod eval;
