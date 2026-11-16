//! S1: the singular-point matrix (owner decision of 2026-11-15).
//!
//! For every inverse function of the series engine (design §2.4, the atan family: arcsin, arccos,
//! arctan, arcsinh, arccosh, arctanh) and the other branch functions the design treats (log, sqrt,
//! the non-integer powers `u^(1/3)`, `u^(2/3)`, `u^(-1/2)`, `u^pi`; §2.4 "Non-integer powers", §2.5
//! `Pow`; the power with a variable exponent, `u^(x+1)` and `u^(2+sin(x))`, which §2.5 expands as
//! `exp(r log u)` for a base that is not the exact zero and by the rule `0^r` for one that is; and
//! the powers with a complex constant exponent, `u^I`, `u^(1/2+I)` and `u^(-1/2+I)`), and for every
//! finite branch point c of each (arcsin, arccos, arccosh, arctanh at +-1; arctan, arcsinh at +-I;
//! log, sqrt and the powers at 0), `data/s1_singular_points.tsv` holds:
//!
//! - (a) arguments whose series is the exact constant c, in several syntactic shapes (c itself,
//!   `c*x^0` at 0 and at 2, `c*(x/x)` at 0, `c*((x-2)/(x-2))` at 2, `c*sign(x^2+1)` at 0 and at
//!   -1/2, `c + 0*x`, c plus the exact zero `(x+1)^2-x^2-2*x-1`, abs of a rational series
//!   `c*abs(-x^0)`, a power with an exact-constant exponent `c*x^(x-x)`, a nested inverse function
//!   `c + arcsin(x-x)`, and `x + f(..)`, `x + 0*f(..)`).
//!   Owner's rule: the value f(c), with no expansion and no integrand. Expected: the value, against
//!   gp's f(c); where f(c) is a pole of the constant (arctan(+-I), arctanh(+-1), log(0), 0^(-1/2),
//!   0^(-1/2+I)) the design's typed answer for a pole of a constant, `DivisionByZero` (§2.2
//!   `apply`; review 9 B1 required change 1 and m2); and for 0^I, which has no value (Re r = 0),
//!   `NotLaurent` (§2.5 `zero_pow_sign`, the third outcome of an exact-constant argument; only the
//!   family `powI`).
//! - (b) non-constant arguments with constant term c (`c + x`, `c - x^2`, `c + x^3`, `c + sin(x)`,
//!   `c*exp(x)` at 0; x at the point c; `c*x^2` at 1; `x + 0*f(c + x)`). Owner's rule:
//!   `NotLaurent`, never `DivisionByZero`.
//! - (c) the regular points c + 1/2 and c - 1/2 next to each branch point: f(x) there, order 3,
//!   against gp's series (points on a branch cut included).
//! - (d) two more shapes of a branch point (S1 implementation review 1, semantics lens, m5): the
//!   branch point at infinity, `f(1/x)` at 0, once per function; and non-constant arguments whose
//!   constant term is c only through an exact kernel value (`sin(pi/2 + x)`, `2*sin(pi/6 + x)`,
//!   `-sin(pi/2 + x)`, `cos(pi + x)`, `+-I*sin(pi/2 + x)`, `+-I*cosh(x)` at 0; `sin(x)` at pi and
//!   `cos(x)` at pi/2 for c = 0). Owner's rule: `NotLaurent`. gp evaluates pi as a float, so it
//!   often returns a series for these rows: its constant term is only near c.
//! - (e) three guard rows (S1 implementation review 2, semantics lens, minor 6), after the (d)
//!   rows: `arcsin(1^(1+x))` (`PrecisionExhausted`: the argument is the constant 1, but its series
//!   is truncated), `arcsin((-1)^(1+x))` (`NotLaurent`: exp(log(-1)) is folded to -1 since fix-up
//!   round 2) and `arcsin(sin(1)^2+cos(1)^2+x)` (`UndecidedZero`: the constant field cannot decide
//!   sin(1)^2 + cos(1)^2 = 1), each at 0. Honest typed refusals, never `DivisionByZero`.
//! - (f) four rows per pair whose f(c) is finite (S1 fix-up round 4, review 4, semantics lens,
//!   m1), after the (e) rows: `1/(3 + f(c x^0))`, `1/(x + 3 + f(c x^0))`, `log(x + 3 + f(c x^0))`
//!   and `sqrt(x + 3 + f(c x^0))` at 0. The owner's first rule asks for a value, and the sum
//!   3 + f(c) must be decided nonzero: the values (gp on the simplified constant).
//!
//! Intended refusals (not regressions in the sense of S1_BAR (b)). On 39 rows the matrix expects a
//! refusal where HEAD's taylor returns x (checked on HEAD: `x + 0*arctan(I*x^0)` is x). They are
//! the rows with a constant factor 0 in front of a subexpression the engine refuses, since the
//! engine expands each subexpression on its own and "a constant factor 0 ... their errors still
//! surface" (design §2.5, §2.9):
//! - the 23 `b:*:sub` rows `x + 0*f(c + x)` (one per family), `NotLaurent`;
//! - the 16 rows `a:*:1+0x` and `a:*:7+0x` (`x + 0*f(c*x^0)` and `x + 0*f(c + 0*x)`) of the eight
//!   families whose f(c) is refused: arctan@I, arctan@-I, arctanh@1, arctanh@-1, log@0,
//!   pow(-1/2)@0 and pow(-1/2+I)@0 (`DivisionByZero`) and powI@0 (`NotLaurent`).
//! The regression file's `x + 0*atanh(1+x)` (`B2_ROWS`) is the same class.
//!
//! Every row goes through `series_expansion` and `taylor_with_budget` with the REPL budget and
//! through the limit path (`Finite` only for a rational constant term, §2.13), on a thread with
//! the bar's stack (2 MB in debug, 8 MB in release). The data and their gp references are made by
//! `data/gen_s1_singular.py` (`python3 -I gen_s1_singular.py > s1_singular_points.tsv` in
//! `tests/data`; gp 2.15.4 at realprecision 300 through `data/s1_gp.py`).
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.5 transcendental functions); Knuth, TAOCP vol. 2, §4.7; Brent and Kung, J. ACM 25(4), 1978.

mod s1_common;

use s1_common::check::{check_all, parse_rows, Kind, Row, RowWant};

const DATA: &str = include_str!("data/s1_singular_points.tsv");
const DIGITS: u32 = 60;
const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

fn rows(prefix: &str) -> Vec<Row> {
    parse_rows(DATA)
        .into_iter()
        .filter(|r| r.id.starts_with(prefix))
        .collect()
}

#[test]
fn matrix_covers_every_function_and_branch_point() {
    let all = parse_rows(DATA);
    for fc in [
        "arcsin@1",
        "arcsin@-1",
        "arccos@1",
        "arccos@-1",
        "arctan@I",
        "arctan@-I",
        "arcsinh@I",
        "arcsinh@-I",
        "arccosh@1",
        "arccosh@-1",
        "arctanh@1",
        "arctanh@-1",
        "log@0",
        "sqrt@0",
        "pow(1/3)@0",
        "pow(2/3)@0",
        "pow(-1/2)@0",
        "powpi@0",
        "pow(x+1)@0",
        "pow(2+sin(x))@0",
        "powI@0",
        "pow(1/2+I)@0",
        "pow(-1/2+I)@0",
    ] {
        for class in ["a:", "b:", "c:"] {
            let n = all
                .iter()
                .filter(|r| r.id.starts_with(&format!("{class}{fc}:")))
                .count();
            assert!(n >= 2, "class {class} of {fc} has {n} rows");
        }
        // (d): two arguments through an exact kernel value, and f(1/x) at 0 for the function
        let n = all
            .iter()
            .filter(|r| r.id.starts_with(&format!("d:{fc}:kid")))
            .count();
        assert!(n >= 2, "class d: of {fc} has {n} rows");
        let f = fc.split('@').next().unwrap();
        assert!(
            all.iter().any(|r| r.id == format!("d:{f}@inf:0")),
            "class d: has no row f(1/x) for {f}"
        );
    }
    // (f), fix-up round 4: four rows for each pair whose f(c) is finite (every (a) family with a
    // value row)
    for fc in [
        "arcsin@1",
        "arcsin@-1",
        "arccos@1",
        "arccos@-1",
        "arcsinh@I",
        "arcsinh@-I",
        "arccosh@1",
        "arccosh@-1",
        "sqrt@0",
        "pow(1/3)@0",
        "pow(2/3)@0",
        "powpi@0",
        "pow(x+1)@0",
        "pow(2+sin(x))@0",
        "pow(1/2+I)@0",
    ] {
        let n = all
            .iter()
            .filter(|r| r.id.starts_with(&format!("f:{fc}:")))
            .count();
        assert_eq!(n, 4, "class f: of {fc} has {n} rows");
    }
    // (g), fix-up round 5: a non-constant and an exact-constant row for each inverse function whose
    // argument reaches a branch point through a folded value, and two rows for each value round 5
    // adds
    for f in [
        "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
    ] {
        for kind in ["nonconst", "exact"] {
            assert!(
                all.iter().any(|r| r.id == format!("g:{f}:{kind}")),
                "class g: has no row {f}:{kind}"
            );
        }
    }
    for fc in [
        "arcsin@1/2",
        "arcsin@-1/2",
        "arccos@1/2",
        "arccos@-1/2",
        "arccosh@1/2",
        "arccosh@-1/2",
        "arcsinh@I/2",
        "arcsinh@-I/2",
        "arctanh@I",
        "arctanh@-I",
    ] {
        for kind in ["inv", "zero"] {
            assert!(
                all.iter().any(|r| r.id == format!("g:{fc}:{kind}")),
                "class g: has no row {fc}:{kind}"
            );
        }
    }
    // (a) is a value or DivisionByZero (a pole of the constant), or NotLaurent for 0^I only (it
    // has no value); (b) and (d) are NotLaurent; (c) and (f) a value
    for r in &all {
        match (&r.id[..2], &r.want) {
            ("a:", RowWant::Value(_)) | ("a:", RowWant::Refusal(Kind::DivisionByZero)) => {}
            ("a:", RowWant::Refusal(Kind::NotLaurent)) if r.id.starts_with("a:powI@0:") => {}
            ("b:", RowWant::Refusal(Kind::NotLaurent)) => {}
            ("c:", RowWant::Value(_)) => {}
            ("d:", RowWant::Refusal(Kind::NotLaurent)) => {}
            ("e:", RowWant::Refusal(k)) if *k != Kind::DivisionByZero => {}
            ("f:", RowWant::Value(_)) => {}
            ("g:", RowWant::Refusal(Kind::NotLaurent)) if r.id.ends_with(":nonconst") => {}
            ("g:", RowWant::Value(_)) if r.id.ends_with(":exact") || r.id.ends_with(":inv") => {}
            ("g:", RowWant::Refusal(Kind::DivisionByZero))
                if r.id.ends_with(":exact") || r.id.ends_with(":zero") => {}
            _ => panic!("row {} has the expectation {:?}", r.id, r.want),
        }
    }
}

/// (a): an exact-constant argument is evaluated, f(c), or `DivisionByZero` at a pole of f
/// (`NotLaurent` for 0^I). Review 9 B1 (revisions 7rt1 to 8 and the prototype proto13 give
/// `DivisionByZero("0 to a negative power")` for arcsin, arccos, arccosh at +-1 and arcsinh at +-I,
/// and `PrecisionExhausted` for the nested shape `c + arcsin(x-x)`, whose inner arcsin of the
/// exact 0 the prototype integrates to a truncated series).
#[test]
fn exact_constant_arguments_are_evaluated() {
    check_all(rows("a:"), DIGITS, STACK);
}

/// (b): a non-constant argument at a branch point is `NotLaurent`, never `DivisionByZero`.
/// Review 9 B2 (proto13 gives `DivisionByZero("arctanh at 1 or -1")` for arctanh at +-1).
#[test]
fn non_constant_arguments_at_branch_points_are_not_laurent() {
    check_all(rows("b:"), DIGITS, STACK);
}

/// (c): the regular points c +- 1/2 give the series, equal to gp's.
#[test]
fn regular_neighbours_match_gp() {
    check_all(rows("c:"), DIGITS, STACK);
}

/// (d): the branch point at infinity (`f(1/x)` at 0) and arguments whose constant term is the
/// branch point through an exact kernel value are `NotLaurent`, never `DivisionByZero`.
#[test]
fn other_shapes_of_branch_points_are_not_laurent() {
    check_all(rows("d:"), DIGITS, STACK);
}

/// (e): guard rows of honest refusals for constant terms the engine cannot read exactly as the
/// branch point (review 2, semantics lens, minor 6): never `DivisionByZero`.
#[test]
fn guard_shapes_are_typed_refusals() {
    let r = rows("e:");
    assert_eq!(r.len(), 3, "the (e) rows");
    check_all(r, DIGITS, STACK);
}

/// (f), S1 fix-up round 4 (review 4, semantics lens, m1): the value f(c) of an exact-constant
/// argument at a branch point is decided in a sum, so it can be inverted, logged and rooted:
/// `1/(3 + f(c x^0))`, `1/(x + 3 + f(c x^0))`, `log(x + 3 + f(c x^0))` and
/// `sqrt(x + 3 + f(c x^0))` at 0 are the values gp gives on the simplified constant. Before the
/// round the kernels arcsin(+-1), arccos(-1), arccosh(-1) and arcsinh(+-I) had no ball enclosure,
/// and these rows were `UndecidedZero`.
#[test]
fn inverted_values_at_branch_points_are_decided() {
    let r = rows("f:");
    assert_eq!(r.len(), 60, "the (f) rows");
    check_all(r, DIGITS, STACK);
}

/// (g), S1 fix-up round 5 (review 5, semantics lens, m1, and the owner's first rule):
/// - for each inverse function, an argument whose constant term is a branch point c only through a
///   value folded in round 4 (2 arcsin(1)/pi = 1, ...): c + x is `NotLaurent`, never
///   `DivisionByZero`, and c itself is the value f(c) or `DivisionByZero` at a pole (review 5
///   checked these outcomes by a numerical oracle; no test pinned them);
/// - the exact values that round 5 adds, arcsin(+-1/2) = +-pi/6, arccos(1/2) = pi/3,
///   arccos(-1/2) = 2 pi/3, arccosh(+-1/2) = i arccos(+-1/2), arcsinh(+-i/2) = +-i pi/6 and
///   arctanh(+-i) = +-i pi/4: `1/(3 + f(c x^0))` at 0 is the value (before the round
///   `UndecidedZero`: the kernels had no enclosure), and `x/(f(c x^0) - value)` divides by an exact
///   zero (`DivisionByZero`; before, the value `x/(f(c) - value)` of a constant factor).
#[test]
fn values_reached_through_folded_constants() {
    let r = rows("g:");
    assert_eq!(r.len(), 32, "the (g) rows");
    check_all(r, DIGITS, STACK);
}
