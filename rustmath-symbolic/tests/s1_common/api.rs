//! The public entry points of the S1 design, in one place.
//!
//! Design §2.9 (`rustmath-symbolic/src/series.rs`): `Expr::series_expansion(var, point,
//! max_exponent, budget) -> Result<SeriesExpansion, SeriesError>`, `SeriesExpansion::{terms,
//! precision, first_exponent, coefficient, to_expr}`, `Expr::taylor_with_budget(var, point, order,
//! budget) -> Result<Expr, SeriesError>`; `lib.rs` exports `SeriesError`, `SeriesExpansion`,
//! `ExpansionPoint`, and `StructureLimit` lives in `series.rs`. Design §2.13
//! (`rustmath-symbolic/src/limits.rs`): `limit_via_series` keeps its signature and runs the engine
//! with `LIMIT_SERIES_WORK = 1e8`; `limit_with_error` keeps its signature and builds the `BigO`
//! from an expansion that renders no term. `limit_enhanced` (one-sided limits) is HEAD's, unchanged.
//!
//! Every test calls these wrappers, so a change of a signature in the implementation is a change
//! of this file only.
pub use rustmath_symbolic::series::StructureLimit;
pub use rustmath_symbolic::{
    BigO, Direction, ExpansionPoint, Expr, LimitResult, SeriesError, SeriesExpansion, Symbol,
};

/// The REPL's budget for one `taylor` line: `rustmath-jupyter/src/repl_limits.rs`
/// `TAYLOR_WORK` (design §3.1 and §5.3, 100_000_000). rustmath-symbolic cannot depend on the REPL
/// crate, so the value is repeated here; keep the two equal.
pub const REPL_TAYLOR_WORK: u64 = 100_000_000;

/// The port's parser (design §2.12).
pub fn parse(s: &str) -> Expr {
    match rustmath_symbolic::parse(s) {
        Ok(e) => e,
        Err(e) => panic!("parse({s:?}) failed: {e}"),
    }
}

/// `taylor_with_budget`.
pub fn tw(f: &Expr, x: &Symbol, a: &Expr, order: usize, budget: u64) -> Result<Expr, SeriesError> {
    f.taylor_with_budget(x, a, order, budget)
}

/// `series_expansion`.
pub fn se(
    f: &Expr,
    x: &Symbol,
    point: &ExpansionPoint,
    max_exponent: i64,
    budget: Option<u64>,
) -> Result<SeriesExpansion, SeriesError> {
    f.series_expansion(x, point, max_exponent, budget)
}

/// The limit path: `limit_via_series` with HEAD's default order 5 (its own budget,
/// `LIMIT_SERIES_WORK`, design §2.13).
pub fn lim(f: &Expr, x: &Symbol, a: &Expr) -> LimitResult {
    f.limit_via_series(x, a, 5)
}

/// A one-sided limit: HEAD's `limit_enhanced` (`limits.rs`), which S1 does not change. Its series
/// strategy calls `limit_via_series` with the order 5 for every direction (design §2.13).
pub fn lim_dir(f: &Expr, x: &Symbol, a: &Expr, dir: Direction) -> LimitResult {
    f.limit_enhanced(x, a, dir)
}

/// `limit_with_error` (two-sided).
pub fn lwe(f: &Expr, x: &Symbol, a: &Expr, order: usize) -> (LimitResult, Option<BigO>) {
    f.limit_with_error(x, a, Direction::Both, order)
}

/// The exponent n of the `BigO` `O(h^n)`: the function is `h^n` (`Pow(h, Integer(n))`) or `h`.
pub fn big_o_exponent(b: &BigO) -> Option<i64> {
    match &b.function {
        Expr::Binary(rustmath_symbolic::BinaryOp::Pow, _, n) => match &**n {
            Expr::Integer(n) => n.to_i64(),
            _ => None,
        },
        Expr::Symbol(_) => Some(1),
        Expr::Binary(rustmath_symbolic::BinaryOp::Sub, _, _) => Some(1),
        _ => None,
    }
}
