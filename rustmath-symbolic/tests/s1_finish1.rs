//! S1 finish, fix-up round 1: four blockers of the reviews fin1, written as regression tests
//! before the fixes. (The parser blockers, semantics B2 and resources B1 and B2, are in
//! `tests/s1_depth.rs`, tests `p6_*` and `f1_*`.)
//!
//! - Semantics B1 (`fu2/s1impl/S1.fin1-semantics.md`), bar (a): f^g with f of decided nonzero
//!   valuation was `NotLaurent`, a decided answer, whenever the exponent's series had a stored
//!   term other than its constant term, also when every such coefficient is undecided and is 0.
//!   The function is then f^k, a Laurent series: `x^(1+(log(2)+log(3)-log(6))*x)` at 0 is x, and
//!   `(x-pi)^((1+x)/(1+x))` at pi is x - pi. Expected now: a refusal that decides nothing
//!   (`UndecidedZero` of the first undecided coefficient, or `PrecisionExhausted`), never
//!   `NotLaurent`, a value or `DivisionByZero`. A stored term with a decided nonzero coefficient,
//!   or a decided non-integer constant term, keeps `NotLaurent`.
//! - Resources B3 (`fu2/s1impl/S1.fin1-resources.md`), bar (c): the re-parse of a printed result
//!   took up to 111 s (32-byte inputs). The parser folds each printed p/q into a `Rational` (a
//!   gcd), and parses each long literal in quadratic time, at every term.
//! - Resources B4, bar (c): the re-parse of a printed result overflowed the stack of a thread on
//!   which its input parsed. A result nests deeper than its input: the sum of its terms, the
//!   product with h^k, and `Display`'s parentheses around every binary node of a kernel's source.
//! - Resources B5, bar (b): `expand` of the parsed signed tower `x^-2^-2^...^-2`, a text HEAD
//!   reads, aborted from 959 levels on a 2 MB debug thread and from 15 410 on an 8 MB release
//!   thread. The port's tree has two nodes per level (`Neg` and `Pow`), and `expand` took a frame
//!   per node. HEAD parses, expands and drops up to 1 439 and 22 772 levels.
//!
//! The minor items of the reviews fin2 (tests `fin2_*`, at the end): the exception to the
//! re-parse of a printed result that the docs now state (semantics m1, resources minor 1), the
//! design's message for a literal zero divisor (resources minor 2) and the message of
//! `PrecisionExhausted` (semantics m3).
//!
//! Reference values: PARI/GP 2.15.4 (`/usr/bin/gp`), `\p 200`, the semantics reviewer's script
//! `fu2/s1impl/fin1-semantics/b1_refs.gp` (run as `gp -q < b1_refs.gp`): `log(2)+log(3)-log(6)`,
//! `atan(1/2)+atan(1/3)-Pi/4`, `sin(1)^2+cos(1)^2-1` and `exp(1)*exp(-1)-1` are 0 to 211 digits,
//! `x^(1+(log(2)+log(3)-log(6))*x) - x` is `0.E-213` on |x| = 1/16, and the control
//! `x^(1+x) - x` is 0.018 there. The identities are exact (log 6 = log 2 + log 3, Machin's
//! formula, sin^2 + cos^2 = 1, e e^-1 = 1).
//!
//! Run: `cargo test -p rustmath-symbolic --test s1_finish1` (debug, 2 MB threads) and
//! `cargo test --release -p rustmath-symbolic --test s1_finish1` (8 MB threads); the budget tests
//! `cargo test --release -p rustmath-symbolic --test s1_finish1 -- --ignored --test-threads=1`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: an O-term is not an exact zero; §49.5: powers of a series); S1 design §2.5
//! (`Pow`), §2.11 (`Display`), §2.12 (the parser), §2.15 (the result caps), §2.17 (`expand`).

mod s1_common;

use rustmath_symbolic::Expr;
use s1_common::api::*;
use s1_common::check::*;
use std::time::Instant;

const MB: usize = 1 << 20;
const DEBUG: bool = cfg!(debug_assertions);
/// The bar's thread: 2 MB in debug, 8 MB in release.
const STACK: usize = if DEBUG { 2 * MB } else { 8 * MB };

/// A named check, run on its own thread with the bar's stack; the failures are collected.
type Case = (String, Box<dyn FnOnce() -> Result<(), String> + Send>);

fn run(cases: Vec<Case>) {
    let mut bad = Vec::new();
    for (name, f) in cases {
        if let Err(m) = on_stack(STACK, f) {
            bad.push(format!("{name}: {m}"));
        }
    }
    assert!(
        bad.is_empty(),
        "{} failures:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

/// The refusals that decide nothing: what an undecided exponent coefficient must give.
const UNDECIDED: [Kind; 2] = [Kind::UndecidedZero, Kind::PrecisionExhausted];

/// A row at a finite point: `series_expansion`, `taylor_with_budget` (REPL budget) and
/// `series_coefficients` each refuse with one of `kinds`, and the limit path is `Unknown`.
fn finite_row(f: &str, a: &str, order: i64, kinds: &'static [Kind]) -> Case {
    let (f, a) = (f.to_string(), a.to_string());
    (
        format!("{f} at {a}"),
        Box::new(move || {
            let outs: Vec<Outcome<'_>> = kinds
                .iter()
                .map(|k| Outcome::Refusal(*k, LimWant::Unknown))
                .collect();
            check_outcomes(&f, &a, order, &outs, 40)?;
            let fe = parse(&f);
            let ae = parse(&a);
            match fe.series_coefficients(&x(), &ae, order as usize) {
                Err(e) if kinds.contains(&kind(&e)) => Ok(()),
                Err(e) => Err(format!(
                    "series_coefficients: expected {kinds:?}, got Err({e:?})"
                )),
                Ok(_) => Err(format!(
                    "series_coefficients: expected {kinds:?}, got a value"
                )),
            }
        }),
    )
}

/// A row at infinity: `series_expansion` (REPL budget) refuses with one of `kinds`.
fn infinity_row(f: &str, order: i64, kinds: &'static [Kind]) -> Case {
    let f = f.to_string();
    (
        format!("{f} at infinity"),
        Box::new(move || {
            let fe = parse(&f);
            match se(
                &fe,
                &x(),
                &ExpansionPoint::Infinity,
                order,
                Some(REPL_TAYLOR_WORK),
            ) {
                Err(e) if kinds.contains(&kind(&e)) => Ok(()),
                Err(e) => Err(format!(
                    "series_expansion: expected {kinds:?}, got Err({e:?})"
                )),
                Ok(s) => Err(format!(
                    "series_expansion: expected {kinds:?}, got Ok({})",
                    s.to_expr()
                )),
            }
        }),
    )
}

/// Semantics B1: the reviewer's table (`out/b1.tsv`). Each exponent has a stored term other than
/// its constant term whose coefficient is undecided and is 0, so the function is f^k (k an
/// integer), a Laurent series. Before: `NotLaurent` on every row, in both profiles.
#[test]
fn b1f1_an_undecided_exponent_coefficient_is_not_decided() {
    let cases = vec![
        // x (gp: x^(1+U*x) - x is 0.E-213 on |x| = 1/16)
        finite_row("x^(1+(log(2)+log(3)-log(6))*x)", "0", 3, &UNDECIDED),
        // 1
        finite_row("x^((log(2)+log(3)-log(6))/x)", "0", 3, &UNDECIDED),
        // x^2
        finite_row("x^(2+(atan(1/2)+atan(1/3)-pi/4)/x^2)", "0", 3, &UNDECIDED),
        // sin(x)^3
        finite_row("sin(x)^(3+(sin(1)^2+cos(1)^2-1)*x)", "0", 3, &UNDECIDED),
        // 1 (gp: (x-1)^((exp(x)*exp(-x)-1)/x^3) - 1 is 9.9E-212 on |x - 1| = 1/16)
        finite_row("(x-1)^((exp(x)*exp(-x)-1)/x^3)", "1", 3, &UNDECIDED),
        // 1
        finite_row("((x-1)^2)^((exp(x)*exp(-x)-1)*(1+x))", "1", 3, &UNDECIDED),
        // review 6's m1 input at a point that is not rational: x - pi, (x - sqrt(2))^2, x - c
        finite_row("(x-pi)^((1+x)/(1+x))", "pi", 3, &UNDECIDED),
        finite_row("(x-sqrt(2))^(2*(1+x)/(1+x))", "sqrt(2)", 3, &UNDECIDED),
        finite_row("(x-1/(pi+I))^((1+x)/(1+x))", "1/(pi+I)", 3, &UNDECIDED),
        // x, at infinity (h = 1/x, the exponent 1 + U/h)
        infinity_row("(1/x)^(1+(log(2)+log(3)-log(6))*x)", 3, &UNDECIDED),
    ];
    run(cases);
}

/// Semantics B1: review 6's m1 shapes at the points where the reviewer's grid (`out/pts.tsv`)
/// found `NotLaurent` (pi, sqrt(2), e, log(2), 2^(1/3), pi/4, asin(1/2)). Every function is f^k,
/// a Laurent series; the contour oracle found every such row Laurent.
#[test]
fn b1f1_m1_shapes_at_irrational_points_are_not_decided() {
    let mut cases = Vec::new();
    for a in [
        "pi",
        "sqrt(2)",
        "e",
        "log(2)",
        "2^(1/3)",
        "pi/4",
        "asin(1/2)",
    ] {
        for shape in ["(x-A)^G", "((x-A)^2)^G", "(1/(x-A))^G"] {
            for g in ["((1+x)/(1+x))", "(2*(1+x)/(1+x))", "((2+x^2)/(2+x^2))"] {
                let f = shape.replace('A', a).replace('G', g);
                cases.push(finite_row(&f, a, 2, &UNDECIDED));
            }
        }
    }
    run(cases);
}

/// Semantics B1, the controls: a stored term with a decided nonzero coefficient, or a decided
/// non-integer constant term, keeps `NotLaurent` (the function has a term h^k log h).
#[test]
fn b1f1_a_decided_exponent_term_keeps_not_laurent() {
    let nl: &'static [Kind] = &[Kind::NotLaurent];
    let cases = vec![
        // the h coefficient 1 + U is decided nonzero
        finite_row("x^(1+x+(log(2)+log(3)-log(6))*x)", "0", 3, nl),
        // U x + x^2: the h^2 coefficient is 1
        finite_row("x^((log(2)+log(3)-log(6))*x+x^2)", "0", 3, nl),
        // a decided non-integer constant term with an undecided h coefficient
        finite_row("x^(1/2+(log(2)+log(3)-log(6))*x)", "0", 3, nl),
        // at pi: the exponent 1 + x has the h coefficient 1
        finite_row("(x-pi)^(1+x)", "pi", 3, nl),
        finite_row("x^(1+x)", "0", 3, nl),
    ];
    run(cases);
}

/// Resources B3: the re-parse of the printed result of `taylor_with_budget` (REPL budget) at 0,
/// on an 8 MB release thread, within the bar (20 s), or the result is refused. Before: q4 at
/// order 100 re-parsed in 18.4 s, at 514 (its largest accepted order) in 95.1 s, and q5 at 282 in
/// 111.0 s (`rp/*.log`). Release only (the times of the bar are release times).
#[test]
#[ignore]
fn b3f1_reparse_of_a_result_within_the_bar() {
    if DEBUG {
        panic!("a release budget test: run with --release");
    }
    for (f, order) in [
        ("(10^100-1)^1050/7^120000/(1-x)", 100usize),
        ("(10^100-1)^1050/7^120000/(1-x)", 514),
        ("(10^150-1)^1050/7^180000/(1-x)", 282),
        ("(10^30-1)^1050/7^37000/(1-x)", 2_192),
    ] {
        let msg = on_stack(8 * MB, move || {
            let e = parse(f);
            match tw(&e, &x(), &Expr::from(0), order, REPL_TAYLOR_WORK) {
                Err(err) => match kind(&err) {
                    Kind::LimitExceeded(_) | Kind::BudgetExceeded => Ok(format!("refused: {err}")),
                    k => Err(format!("{f} at order {order}: an unexpected refusal {k:?}")),
                },
                Ok(r) => {
                    let text = r.to_string();
                    let t = Instant::now();
                    let back = rustmath_symbolic::parse(&text);
                    let secs = t.elapsed().as_secs_f64();
                    drop(back);
                    if secs > BAR_SECONDS {
                        Err(format!(
                            "{f} at order {order}: the re-parse of {} bytes took {secs:.1} s",
                            text.len()
                        ))
                    } else {
                        Ok(format!("re-parsed {} bytes in {secs:.2} s", text.len()))
                    }
                }
            }
        });
        match msg {
            Ok(m) => eprintln!("{f} at order {order}: {m}"),
            Err(m) => panic!("{m}"),
        }
    }
}

/// `sin(sin(...sin(1)...))` with n calls.
fn sin_chain(n: usize) -> String {
    format!("{}1{}", "sin(".repeat(n), ")".repeat(n))
}

/// The printed result of `taylor_with_budget` (REPL budget) of `f` at 0, re-parsed on the same
/// thread on which `f` parsed: `Ok` or a `ParseError` (a stack overflow aborts this test binary),
/// or the result is refused by a structural limit. Returns what happened.
fn reparse_or_refused(f: String, order: usize, stack: usize) -> Result<String, String> {
    on_stack(stack, move || {
        let e = rustmath_symbolic::parse(&f).map_err(|e| format!("the input: {e}"))?;
        match tw(&e, &x(), &Expr::from(0), order, REPL_TAYLOR_WORK) {
            Err(err) => match kind(&err) {
                Kind::LimitExceeded(_) => Ok(format!("refused: {err}")),
                k => Err(format!("an unexpected refusal {k:?}")),
            },
            Ok(r) => {
                let text = r.to_string();
                let back = rustmath_symbolic::parse(&text);
                Ok(format!(
                    "re-parsed {} bytes: {}",
                    text.len(),
                    if back.is_ok() { "Ok" } else { "ParseError" }
                ))
            }
        }
    })
}

/// Resources B4: the reviewer's rows. Debug, 2 MB: `sin^195(1)/(1-x)` (982 bytes) at order 1000
/// parses and gives a result whose text nests 207 levels against the input's 195, and its
/// re-parse aborted. Release, 8 MB: `sin^4320(1)/(1-x)` (21.6 KB) at order 100 (4 329 levels
/// against 4 320).
#[test]
fn b4f1_reparse_of_a_result_does_not_overflow() {
    let rows: Vec<(String, usize)> = if DEBUG {
        vec![(format!("{}/(1-x)", sin_chain(195)), 1_000)]
    } else {
        vec![
            (format!("{}/(1-x)", sin_chain(4_320)), 100),
            (format!("{}/(1-x)", sin_chain(4_320)), 400),
        ]
    };
    for (f, order) in rows {
        let name = format!("sin^n(1)/(1-x), {} bytes, order {order}", f.len());
        match reparse_or_refused(f, order, STACK) {
            Ok(m) => eprintln!("{name}: {m}"),
            Err(m) => panic!("{name}: {m}"),
        }
    }
}

/// Resources B4, the same class through `Display`'s parentheses: a flat sum inside a call parses
/// at the depth of the call, but its tree is a left-deep chain of `+`, which `Display` writes
/// with a parenthesis per node. The result names the input's kernel at every term, so its text
/// nests about as many levels as the sum has terms.
#[test]
fn b4f1_reparse_of_a_kernel_with_a_flat_sum_does_not_overflow() {
    let n = if DEBUG { 300 } else { 4_500 };
    let sum: Vec<String> = (2..n + 2).map(|k| format!("sqrt({k})")).collect();
    let f = format!("sin({})/(1-x)", sum.join("+"));
    match reparse_or_refused(f, 8, STACK) {
        Ok(m) => eprintln!("sin(sqrt(2)+...)/(1-x), {n} terms: {m}"),
        Err(m) => panic!("{m}"),
    }
}

/// Resources B4, the fix: the parser runs on a heap stack, so 100 KB of nesting in HEAD's syntax
/// (33 333 nested parentheses or calls, 50 000 carets, 25 000 levels of `2^(`) parses on a 2 MB
/// thread in both profiles, where the recursive parser stopped at 240 parentheses (debug) and
/// 1 079 (release). The trees are dropped by `Expr`'s iterative `Drop`.
#[test]
fn b4f1_the_parser_takes_no_stack_per_level() {
    let n = 100_000;
    for (open, close, per) in [
        ("(", ")", 2),
        ("sin(", ")", 5),
        ("2^(", ")", 4),
        ("-(", ")", 3),
        ("f(x,", ")", 5),
    ] {
        let k = n / per;
        let text = format!("{}x{}", open.repeat(k), close.repeat(k));
        let r = on_stack(2 * MB, move || rustmath_symbolic::parse(&text).is_ok());
        assert!(r, "{open:?} nested {k} times on a 2 MB thread");
    }
    let tower = vec!["x"; 50_001].join("^");
    let r = on_stack(2 * MB, move || rustmath_symbolic::parse(&tower).is_ok());
    assert!(r, "50 000 carets on a 2 MB thread");
}

/// Resources B4, the control: a result well inside the parser's reach is a value and re-parses.
#[test]
fn b4f1_a_moderate_result_reparses() {
    let f = format!("{}/(1-x)", sin_chain(if DEBUG { 100 } else { 2_000 }));
    let r = on_stack(STACK, move || {
        let e = parse(&f);
        let r = tw(&e, &x(), &Expr::from(0), 200, REPL_TAYLOR_WORK)
            .map_err(|e| format!("taylor_with_budget: {e:?}"))?;
        rustmath_symbolic::parse(&r.to_string())
            .map(|_| ())
            .map_err(|e| format!("re-parse: {e}"))
    });
    assert!(r.is_ok(), "{r:?}");
}

/// Resources B5: `x` followed by n copies of `step`, with an optional prefix.
fn tower(prefix: &str, step: &str, n: usize) -> String {
    format!("{prefix}x{}", step.repeat(n))
}

/// Resources B5: the signed towers that HEAD parses, expands and drops on the bar's threads
/// (HEAD's parse reach: 1 439 levels on 2 MB debug, 22 772 on 8 MB release; with white space
/// 19 996 in release). The port parses them, and `expand`, which leaves them unchanged (no product
/// and no integer power to expand), completes; both trees are dropped. Before: an abort from 959
/// levels (debug) and from 15 410 (release).
#[test]
fn b5f1_expand_of_a_parsed_signed_tower() {
    let rows: Vec<(&str, &str, usize)> = if DEBUG {
        vec![
            ("", "^-2", 1_439),
            ("-", "^-2", 1_439),
            ("", "^-2.5", 1_439),
            ("", " ^ -2", 1_439),
        ]
    } else {
        vec![
            ("", "^-2", 22_772),
            ("-", "^-2", 22_772),
            ("", "^-2.5", 22_770),
            ("", " ^ -2", 19_996),
        ]
    };
    for (prefix, step, n) in rows {
        let text = tower(prefix, step, n);
        let same = on_stack(STACK, move || {
            let e = rustmath_symbolic::parse(&text).expect("HEAD's syntax parses");
            let ex = e.expand();
            // Display is iterative (S1 design §2.11); `==` would recurse as deep as the tree
            let same = e.to_string() == ex.to_string();
            drop(ex);
            drop(e);
            same
        });
        assert!(
            same,
            "{prefix}x{step}... of {n} levels: expand changed the tree"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// S1 finish, minor items of the reviews fin2 (`fu2/s1impl/S1.fin2-semantics.md` m1 to m3,
// `fu2/s1impl/S1.fin2-resources.md` minors 1 to 3). Written before the changes.

/// The refusal of the parser's depth limit (`parser.rs`, `mod depth`).
const TOO_DEEP: &str = "the expression nests too deeply for its syntax";

/// `A1+...+An` with the 30-digit integers Ak = 10^29 + k, each above i64::MAX.
fn big_sum(n: usize) -> String {
    let terms: Vec<String> = (1..=n).map(|k| format!("1{:029}", k)).collect();
    terms.join("+")
}

/// The printed result of `taylor_with_budget` (REPL budget) of `f` at `a`, re-parsed on the
/// same thread: `Ok(())`, or the message of the `ParseError`.
fn reparse_message(f: String, a: i64, order: usize) -> Result<(), String> {
    on_stack(STACK, move || {
        let e = parse(&f);
        let r = tw(&e, &x(), &Expr::from(a), order, REPL_TAYLOR_WORK)
            .unwrap_or_else(|err| panic!("taylor_with_budget of {f}: {err:?}"));
        rustmath_symbolic::parse(&r.to_string())
            .map(|_| ())
            .map_err(|e| e.message)
    })
}

/// Semantics m1 and resources minor 1 (no change of behaviour; the docs now state it): a printed
/// result whose text holds a construct that relaxes the parser's depth limit (an integer above
/// i64::MAX, here 1/21! and beyond, or `^ -(`, which is how `Display` writes the negated
/// exponent the parser builds for HEAD's `x^-2^-2`) and then nests more than 128 levels of
/// parentheses and calls, or 1 024 units in all, is refused by the limit at once when it is
/// re-parsed: a typed `ParseError`, not a crash. The reviewers' examples, with the controls just
/// below the edge. The doc of `mod depth` and of `taylor_with_budget` cite these rows; a
/// follow-up that exempts the parser's own printed forms changes the refused rows to `Ok`.
#[test]
fn fin2_a_printed_relaxing_construct_limits_the_reparse() {
    let tower = |n: usize| format!("x{}", "^-2".repeat(n));
    let rows: Vec<(String, i64, usize, bool)> = vec![
        // 302 and 332 bytes, HEAD's syntax, at 1, order 3
        (tower(100), 1, 3, true),
        (tower(110), 1, 3, false),
        // 100 000 bytes, HEAD's syntax (the fin2 resources review: both results refused)
        (tower(33_333), 1, 3, false),
        // 767 bytes; 20! <= i64::MAX < 21!
        (format!("exp(x) + {}/(1-x)", sin_chain(150)), 0, 20, true),
        (format!("exp(x) + {}/(1-x)", sin_chain(150)), 0, 21, false),
        // 659 bytes, HEAD's syntax
        (format!("exp(x)*{}", sin_chain(130)), 0, 20, true),
        (format!("exp(x)*{}", sin_chain(130)), 0, 25, false),
        // a flat sum of 30-digit integers in a kernel: `Display` writes the kernel's source, a
        // left-deep chain, with a parenthesis per `+`, so the first literal comes after them
        (format!("sin({})/(1-x)", big_sum(100)), 0, 3, true),
        (format!("sin({})/(1-x)", big_sum(200)), 0, 3, false),
    ];
    let mut bad = Vec::new();
    for (f, a, order, reparses) in rows {
        let name = format!("{} bytes at {a}, order {order}", f.len());
        match (reparse_message(f, a, order), reparses) {
            (Ok(()), true) => {}
            (Err(m), false) if m.starts_with(TOO_DEEP) => {}
            (got, _) => bad.push(format!("{name}: expected re-parse {reparses}, got {got:?}")),
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Resources minor 2: a quotient of two integer literals with a literal zero divisor is refused
/// with the design's message (§2.12: "division by zero in a literal"), not nom's generic
/// "Parsing Failure: Error { input: \"\", code: Verify }". HEAD refused `1/0` and read `5 / 0`,
/// `(1)/(0)` and `10\r/0` as Div(n, 0); the port folds every quotient of two signed integer
/// literals (the frozen design), so it refuses all of them. A zero divisor that is not a quotient
/// of two literals stays a division node (`x/0`, `5/(0*x)`), as at HEAD for `x/0`.
#[test]
fn fin2_a_literal_zero_divisor_has_the_design_message() {
    for text in [
        "1/0",
        "5 / 0",
        "(1)/(0)",
        "10\r/0",
        "2/-0",
        "x + 3/(0)",
        "(7/7)/-(0/5)",
        "0009223372036854775807 /0",
        "sin(1/0)",
    ] {
        match rustmath_symbolic::parse(text) {
            Ok(e) => panic!("{text:?} parsed to {e}"),
            Err(e) => assert_eq!(e.message, "division by zero in a literal", "{text:?}"),
        }
    }
    for text in ["x/0", "5/(0*x)", "0/5"] {
        assert!(rustmath_symbolic::parse(text).is_ok(), "{text:?}");
    }
}

/// Semantics m3: the message of `PrecisionExhausted` names the bounded search for every case
/// that returns it, not a denominator only: a truncated integer exponent (`x^((1+x)/(1+x))`,
/// `truncated_integer_exponent`), a truncated constant at a branch point (class (h) of the
/// singular-point matrix, `arcsin((1+x)/(1+x))`) and a denominator that vanishes to the working
/// precision (`1/(sin(x)^2+cos(x)^2-1)`).
#[test]
fn fin2_precision_exhausted_names_the_bounded_search() {
    for f in [
        "x^((1+x)/(1+x))",
        "arcsin((1+x)/(1+x))",
        "1/(sin(x)^2+cos(x)^2-1)",
    ] {
        let f = f.to_string();
        let m = on_stack(STACK, move || {
            let e = parse(&f);
            let at0 = ExpansionPoint::At(Expr::from(0));
            match se(&e, &x(), &at0, 2, Some(REPL_TAYLOR_WORK)) {
                Err(err @ SeriesError::PrecisionExhausted { .. }) => err.to_string(),
                other => panic!("{f}: expected PrecisionExhausted, got {other:?}"),
            }
        });
        assert!(
            m.contains("could not decide a coefficient, an exponent or a branch-point argument")
                && m.contains("working precision")
                && !m.contains("denominator vanished"),
            "{m}"
        );
    }
}
