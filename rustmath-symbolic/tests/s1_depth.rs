//! S1 implementation, fix-up round 1: the three blockers of review 1 (resources lens,
//! `fu2/s1impl/S1.review1-resources.md`), written as regression tests before the fixes.
//!
//! - B1, the parser (`parser.rs` `power` and `atom`): in release `atom` was inlined into the
//!   recursive `power`, so each `^` level took 552 bytes of stack against HEAD's 312. HEAD parses
//!   `x^x^...^x` with 22 770 carets on an 8 MB release thread and 5 677 on a 2 MB one; the port
//!   aborted from 13 783 and from about 3 500.
//! - B2, `limit_with_error` (`limits.rs`): it ran HEAD's unbudgeted `Expr::limit` before the
//!   budgeted expansion. `x+x+...+x` of 3 000 terms aborted a 2 MB debug thread and of 30 000 terms
//!   an 8 MB release thread; `x+sin(1)+...+sin(2399)` used 797 MB in release; `x^-x^-...^-x` of
//!   1 400 levels aborted a 2 MB debug thread and of 13 000 levels an 8 MB release thread. Design
//!   §2.13 computes the limit as `limit_via_series` does (the engine under `LIMIT_SERIES_WORK`).
//! - B3, `simplify` (`simplify.rs`): the left-spine loop of design §2.10 made the recursive frame
//!   larger (312 bytes against HEAD's 208 in release). HEAD simplifies a right-nested sum, a tower
//!   `x^(x^(...))` and a `sin` chain of depth 32 640 on an 8 MB release thread (8 169 on 2 MB),
//!   and `Expr::limit` of a tower of depth 29 056 (7 261 on 2 MB); the port aborted from 22 720.
//!   In debug on 2 MB HEAD simplifies a `sin` chain of depth 3 248; the port aborted from 2 416.
//! - Review 2 (`fu2/s1impl/S1.review2-resources.md`), B1: round 1's fix of B3 kept `simplify`'s
//!   frame at HEAD's size but added a frame per level on function calls (`simplify_function`) and
//!   on zigzags (`simplify_left_spine`), so `f(f(...))`, `f(x + f(...))` and `L = (x + L) + x`
//!   aborted below HEAD's depth, in `simplify` and in `Expr::limit` (tests `b1r2_*`).
//!
//! HEAD's limits were measured on HEAD (8a0b310) built by cargo: in release by review 1 and in this
//! round (`fu2/s1impl/fix1/head_depth.py` with review 1's HEAD probes); in debug by review 1
//! (`simplify`, `Expr::limit`) and by the S1 implementation step (the parser, HEAD's `parser.rs`
//! compiled into a cargo test binary: `(` 216, `sin(` 162, carets 1 440, `2^(` 187 on 2 MB). Each
//! test runs at about 96 % of HEAD's limit, so a frame larger than HEAD's by 16 bytes or more
//! fails here, while the build-to-build noise of a few levels does not. A failure is an abort of
//! this test binary (`has overflowed its stack`), so this file holds only depth and budget tests.
//!
//! Run: `cargo test -p rustmath-symbolic --test s1_depth` (debug: 2 MB threads) and
//! `cargo test --release -p rustmath-symbolic --test s1_depth` (release: 8 MB and 2 MB threads).
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`;
//! the series engine's callers: the parser of its input, `simplify`, the limit path); S1 design
//! §2.10, §2.12, §2.13, §2.16 and S1_BAR (b) and (c).

mod s1_common;

use rustmath_symbolic::{BinaryOp, UnaryOp};
use s1_common::api::*;
use s1_common::check::{on_stack, vm_hwm_bytes, x, BAR_BYTES, BAR_SECONDS};
use std::sync::Arc;
use std::time::Instant;

const MB: usize = 1 << 20;
const DEBUG: bool = cfg!(debug_assertions);

/// `x^x^...^x` with n carets.
fn carets(n: usize) -> String {
    vec!["x"; n + 1].join("^")
}

/// `open` repeated n times, then `x`, then n closing parentheses.
fn nested(open: &str, n: usize) -> String {
    format!("{}x{}", open.repeat(n), ")".repeat(n))
}

/// The depth of the right spine of a `Pow` tower.
fn pow_spine(e: &Expr) -> usize {
    let mut d = 0;
    let mut cur = e;
    while let Expr::Binary(BinaryOp::Pow, _, r) = cur {
        d += 1;
        cur = r;
    }
    d
}

/// The number of nested `(` or calls an input reads: the depth of its tree, by an explicit stack.
fn tree_depth(e: &Expr) -> usize {
    let mut best = 0;
    let mut st: Vec<(&Expr, usize)> = vec![(e, 1)];
    while let Some((n, d)) = st.pop() {
        best = best.max(d);
        match n {
            Expr::Binary(_, l, r) => {
                st.push((l, d + 1));
                st.push((r, d + 1));
            }
            Expr::Unary(_, a) => st.push((a, d + 1)),
            Expr::Function(_, a) => st.extend(a.iter().map(|c| (&**c, d + 1))),
            _ => {}
        }
    }
    best
}

// ------------------------------------------------------------------------------------------------
// B1: the parser's recursion is at least as deep as HEAD's
// ------------------------------------------------------------------------------------------------

/// B1: a caret tower parses as deep as HEAD's parser reads it, on each thread of the bar.
/// HEAD: 1 440 carets on 2 MB debug; 22 770 on 8 MB release and 5 677 on 2 MB release.
#[test]
fn b1_caret_tower_depth_not_below_head() {
    let cases: &[(usize, usize)] = if DEBUG {
        &[(2 * MB, 1_380)]
    } else {
        &[(8 * MB, 21_800), (2 * MB, 5_450)]
    };
    for &(stack, n) in cases {
        let d = on_stack(stack, move || {
            let e = rustmath_symbolic::parse(&carets(n)).expect("the tower parses");
            pow_spine(&e)
        });
        assert_eq!(d, n, "{n} carets on a {} MB thread", stack / MB);
    }
}

/// B1: nested parentheses, calls and powers of a parenthesis parse as deep as HEAD's parser reads
/// them. HEAD on 2 MB debug: `(` 216, `sin(` 162, `2^(` 187, `2*(` and `-(` 216; on 8 MB release:
/// `(`, `sin(`, `2*(`, `-(` 4 124 and `2^(` 3 491; on 2 MB release: `(` and `sin(` 1 028, `2^(` 870.
#[test]
fn b1_nesting_depth_not_below_head() {
    let cases: &[(usize, &str, usize)] = if DEBUG {
        &[
            (2 * MB, "(", 207),
            (2 * MB, "sin(", 155),
            (2 * MB, "2^(", 179),
            (2 * MB, "2*(", 207),
            (2 * MB, "-(", 207),
        ]
    } else {
        &[
            (8 * MB, "(", 3_950),
            (8 * MB, "sin(", 3_950),
            (8 * MB, "2^(", 3_350),
            (8 * MB, "2*(", 3_950),
            (8 * MB, "-(", 3_950),
            (2 * MB, "(", 985),
            (2 * MB, "sin(", 985),
            (2 * MB, "2^(", 835),
        ]
    };
    for &(stack, open, n) in cases {
        let d = on_stack(stack, move || {
            let e = rustmath_symbolic::parse(&nested(open, n)).expect("the input parses");
            tree_depth(&e)
        });
        // parentheses make no node: `((x))` is x; each other level makes one
        let want = if open == "(" { 1 } else { n + 1 };
        assert_eq!(
            d,
            want,
            "{open:?} nested {n} times on a {} MB thread",
            stack / MB
        );
    }
}

// ------------------------------------------------------------------------------------------------
// S1 finish (review 6, resources minor 2): a text that HEAD's parser refuses is refused, never an
// abort
// ------------------------------------------------------------------------------------------------

/// `parse` of `text` on a thread of `stack` bytes: the depth of the tree, or the parse error. A
/// stack overflow aborts this test binary.
fn parse_on(stack: usize, text: String) -> Result<usize, String> {
    on_stack(stack, move || {
        rustmath_symbolic::parse(&text)
            .map(|e| tree_depth(&e))
            .map_err(|e| e.to_string())
    })
}

/// The texts of the minor (and of its class), each refused by HEAD's parser at once with a
/// `ParseError`: HEAD reads a sign after `^` only as part of a literal (`x^-2`), at most one sign
/// in front of an operand that is not a literal, and integer literals up to i64::MAX. The port
/// reads them all, and then recursed into the rest of the text, so each of these aborted a 2 MB
/// debug thread (checked with review 6's probes against HEAD's and the port's debug rlibs,
/// `fu2/s1impl/fin/parse/`). Now each is refused with a `ParseError`, on every thread of the bar.
fn refused_by_head(n: usize) -> Vec<(&'static str, String)> {
    let par = format!("{}x{}", "(".repeat(300), ")".repeat(300));
    vec![
        ("x^-x^-...^-x", vec!["x"; n + 1].join("^-")),
        ("-x^-x^-...^-x", format!("-{}", vec!["x"; n + 1].join("^-"))),
        ("x^-x^-... of 100 KB", vec!["x"; 33_000].join("^-")),
        ("x^-(x^-(...))", nested("x^-(", 300)),
        ("--(--(...))", nested("--(", 300)),
        ("x^-x + 300 parentheses", format!("x^-x + {par}")),
        ("x^+2 + 300 parentheses", format!("x^+2 + {par}")),
        ("- - -x + 300 parentheses", format!("- - -x + {par}")),
        (
            "a 20-digit literal + 300 parentheses",
            format!("99999999999999999999 + {par}"),
        ),
    ]
}

/// The minor: `x^-x^-...` and `-x^-x^-...` from 1 500 levels, which HEAD refuses with a
/// `ParseError` (a sign after `^`), aborted the port's parser on a 2 MB debug thread (S1_BAR (b),
/// a regression against HEAD on an input it handled). Each text of the class is now refused, on
/// a 2 MB thread in both profiles and on the release 8 MB thread; the same towers of 500 levels
/// still parse.
#[test]
fn p6_texts_refused_by_head_are_refused_not_aborted() {
    let stacks: &[usize] = if DEBUG { &[2 * MB] } else { &[2 * MB, 8 * MB] };
    for &stack in stacks {
        for (name, text) in refused_by_head(1_500) {
            let r = parse_on(stack, text);
            assert!(r.is_err(), "{name} on a {} MB thread: {r:?}", stack / MB);
        }
        for (name, text) in [
            ("x^-x^-... 500 levels", vec!["x"; 501].join("^-")),
            (
                "-x^-x^-... 500 levels",
                format!("-{}", vec!["x"; 501].join("^-")),
            ),
            ("x^-(x^-(...)) 60 levels", nested("x^-(", 60)),
            ("--(--(...)) 100 levels", nested("--(", 100)),
        ] {
            let r = parse_on(stack, text);
            assert!(r.is_ok(), "{name} on a {} MB thread: {r:?}", stack / MB);
        }
    }
}

/// The control: texts that HEAD's parser reads keep HEAD's reach. HEAD accepts `x^-2`, `--5`
/// (a sign, then the literal -5), `- -5`, `+-5`, `x^ -5.5` and the literal i64::MAX, each before
/// 300 nested parentheses, on the REPL's 64 MiB thread (and overflows on a 2 MB debug thread, as
/// the port does: 300 parentheses are past both parsers' reach there). The port parses them.
#[test]
fn p6_texts_read_by_head_keep_their_reach() {
    let par = format!("{}x{}", "(".repeat(300), ")".repeat(300));
    for head in [
        "x^-2",
        "--5",
        "- -5",
        "+-5",
        "x^ -5.5",
        "9223372036854775807",
    ] {
        let r = parse_on(64 * MB, format!("{head} + {par}"));
        assert!(r.is_ok(), "{head} + 300 parentheses: {r:?}");
    }
    // a sign run, a signed exponent or a long literal after the deep part does not limit it
    for tail in ["x^-x", "---x", "99999999999999999999"] {
        let r = parse_on(64 * MB, format!("{par} + {tail}"));
        assert!(r.is_ok(), "300 parentheses + {tail}: {r:?}");
    }
    // S1 finish, fix-up round 1 (fin1 semantics B2): HEAD read -9223372036854775808 (i64::MIN)
    // as a literal after `^` and as the second sign of an operand; the six texts of the review, at
    // 150 levels, parse on a 2 MB thread in both profiles (HEAD's debug parser reads them at 150
    // and 200 levels there). Before: a `ParseError` (the literal turned the depth limit on)
    let nest = |inner: &str| format!("{}{inner}{}", "(".repeat(150), ")".repeat(150));
    for text in [
        format!("x^-9223372036854775808 + {}", nest("x")),
        format!("--9223372036854775808 + {}", nest("x")),
        format!("+-9223372036854775808 + {}", nest("x")),
        format!("2*--9223372036854775808*{}", nest("y")),
        nest("x^-9223372036854775808 + y"),
        format!("x^-09223372036854775808 + {}", nest("x")),
    ] {
        let short: String = text.chars().take(40).collect();
        let r = parse_on(2 * MB, text);
        assert!(r.is_ok(), "{short}... (150 levels): {r:?}");
    }
    // HEAD refused one more than i64::MIN in magnitude
    let r = parse_on(2 * MB, format!("x^-9223372036854775809 + {}", nest("x")));
    assert!(r.is_err(), "x^-9223372036854775809 + 150 levels: {r:?}");
}

/// The tree's `Integer` leaves, by an explicit stack.
fn integer_leaves(e: &Expr) -> Vec<rustmath_integers::Integer> {
    let mut out = Vec::new();
    let mut st: Vec<&Expr> = vec![e];
    while let Some(n) = st.pop() {
        match n {
            Expr::Integer(i) => out.push(i.clone()),
            Expr::Binary(_, l, r) => {
                st.push(l);
                st.push(r);
            }
            Expr::Unary(_, a) => st.push(a),
            Expr::Function(_, a) => st.extend(a.iter().map(|c| &**c)),
            _ => {}
        }
    }
    out
}

/// S1 finish, fix-up round 1 (fin1 resources B1): HEAD's negative literal i64::MIN keeps HEAD's
/// reach after it: `x^-N*`, `--N*` and `2/-N*` (N = 9223372036854775808) before n parentheses,
/// `x^-N*` before n nested calls and before a tower of n carets. n = 200 parentheses (HEAD reads
/// 215 on 2 MB debug), 150 calls (HEAD: 162) and 1 000 carets (HEAD: 1 438) in debug; 1 000 of each
/// on 8 MB in release (HEAD: 4 124 parentheses and calls, 22 772 carets). Before: refused past 128
/// levels (or 1 024 carets). The exponent keeps the literal: `Integer(-9223372036854775808)`.
/// Controls: `-N*` and `N*` before 300 parentheses stay a `ParseError` (HEAD refused the literal).
#[test]
fn f1_head_negative_literal_keeps_head_reach() {
    let (stack, np, nc, nt) = if DEBUG {
        (2 * MB, 200, 150, 1_000)
    } else {
        (8 * MB, 1_000, 1_000, 1_000)
    };
    let par = |n: usize| format!("{}x{}", "(".repeat(n), ")".repeat(n));
    let n = "9223372036854775808";
    let min = rustmath_integers::Integer::from(i64::MIN);
    for (name, text, has_min) in [
        ("x^-N* parentheses", format!("x^-{n}*{}", par(np)), true),
        ("--N* parentheses", format!("--{n}*{}", par(np)), false),
        ("2/-N* parentheses", format!("2/-{n}*{}", par(np)), false),
        (
            "x^-N* calls",
            format!("x^-{n}*{}", nested("sin(", nc)),
            true,
        ),
        ("x^-N* carets", format!("x^-{n}*{}", carets(nt)), true),
    ] {
        let min = min.clone();
        let r = on_stack(stack, move || {
            rustmath_symbolic::parse(&text)
                .map(|e| integer_leaves(&e).contains(&min))
                .map_err(|e| e.to_string())
        });
        match r {
            Ok(found) => assert!(
                !has_min || found,
                "{name}: no Integer(i64::MIN) in the tree"
            ),
            Err(e) => panic!("{name}: {e}"),
        }
    }
    for text in [format!("-{n}*{}", par(300)), format!("{n}*{}", par(300))] {
        let short: String = text.chars().take(24).collect();
        let r = parse_on(2 * MB, text);
        assert!(r.is_err(), "{short}... + 300 parentheses: {r:?}");
    }
}

/// S1 finish, fix-up round 1 (fin1 resources B2): HEAD's literal rational is an integer, `/` with
/// no white space, and an integer (an optional `-`, digits, a value in i64, not 0). HEAD refused
/// `2/x`, `1/(x+1)`, `1/sin(x)`, `2/ 3`, `2/3.5`, `2/--3`, `2/+3` and `x^2/x` at once (its literal
/// rule failed, and so did the atom). The port reads them, and then read the deep part after
/// them, so each row aborted the thread (2 MB debug: 300 parentheses, 300 calls or 1 601 carets;
/// 8 MB release: 5 000 parentheses or 30 001 carets). Now each gives a `ParseError`, on a 2 MB
/// thread in both profiles and on the 8 MB release thread.
#[test]
fn f1_literal_rationals_refused_by_head_are_refused_not_aborted() {
    let par = |n: usize| format!("{}x{}", "(".repeat(n), ")".repeat(n));
    let mut rows: Vec<(usize, String)> = vec![
        (2 * MB, format!("2/x*{}", par(300))),
        (2 * MB, format!("1/(x+1)*{}", par(300))),
        (2 * MB, format!("2/ 3*{}", par(300))),
        (2 * MB, format!("2/3.5*{}", par(300))),
        (2 * MB, format!("2/--3*{}", par(300))),
        (2 * MB, format!("2/+3*{}", par(300))),
        (2 * MB, format!("x^2/x*{}", par(300))),
        (2 * MB, format!("2 /3/x*{}", par(300))),
        (2 * MB, format!("1/sin(x)*{}", nested("sin(", 300))),
        (2 * MB, format!("2/x*{}", carets(1_601))),
    ];
    if !DEBUG {
        rows.push((8 * MB, format!("2/x*{}", par(5_000))));
        rows.push((8 * MB, format!("2/x*{}", carets(30_001))));
    }
    for (stack, text) in rows {
        let short: String = text.chars().take(16).collect();
        let r = parse_on(stack, text);
        assert!(
            r.is_err(),
            "{short}... on a {} MB thread: {r:?}",
            stack / MB
        );
    }
}

/// The control of `f1_literal_rationals_refused_by_head_are_refused_not_aborted`: texts that HEAD
/// reads keep their reach. `2 / x` is a division (white space), `2/3^x` is HEAD's (2/3)^x,
/// `2/3/x` is (2/3)/x, `2/-N` with N = 2^63 is HEAD's rational 2/i64::MIN, and `x^-N/3` is HEAD's
/// x^(i64::MIN/3); each before 300 parentheses, on the REPL's 64 MiB thread.
#[test]
fn f1_literal_rationals_read_by_head_keep_their_reach() {
    let par = format!("{}x{}", "(".repeat(300), ")".repeat(300));
    for head in [
        "2 / x",
        "2/3^x",
        "2/3/x",
        "2/-9223372036854775808",
        "x^-9223372036854775808/3",
        "6/3/2/1",
        "2/-3",
    ] {
        let r = parse_on(64 * MB, format!("{head}*{par}"));
        assert!(r.is_ok(), "{head} * 300 parentheses: {r:?}");
    }
}

// ------------------------------------------------------------------------------------------------
// B3: simplify (and Expr::limit, which simplifies) recurse at least as deep as HEAD's
// ------------------------------------------------------------------------------------------------

/// A programmatic tree of n levels (no parser), built on a large thread: `x + (x + (...))`
/// (`rsum`), `x^(x^(...))` (`pow`), `sin(sin(...x))` (`sin`), and, for review 2's B1,
/// `f(f(...f(x)))` (`fch`, one level per call), `f(x + f(x + (...)))` (`fadd`, two levels per
/// step) and the zigzag `L = (x + L) + x` (`zig`, two levels per step), as review 2's
/// `consume2.rs` builds them.
fn deep(kind: &'static str, n: usize) -> Expr {
    on_stack(256 * MB, move || {
        let x = || Arc::new(Expr::Symbol(rustmath_symbolic::Symbol::new("x")));
        let add = |a: Arc<Expr>, b: Arc<Expr>| Arc::new(Expr::Binary(BinaryOp::Add, a, b));
        let mut f = Expr::Symbol(rustmath_symbolic::Symbol::new("x"));
        for _ in 0..n {
            f = match kind {
                "rsum" => Expr::Binary(BinaryOp::Add, x(), Arc::new(f)),
                "pow" => Expr::Binary(BinaryOp::Pow, x(), Arc::new(f)),
                "sin" => Expr::Unary(UnaryOp::Sin, Arc::new(f)),
                "fch" => Expr::Function("f".into(), vec![Arc::new(f)]),
                "fadd" => Expr::Function("f".into(), vec![add(x(), Arc::new(f))]),
                "zig" => Expr::Binary(BinaryOp::Add, add(x(), Arc::new(f)), x()),
                _ => unreachable!(),
            };
        }
        f
    })
}

/// The depth of `deep(kind, n)`.
fn deep_depth(kind: &str, n: usize) -> usize {
    match kind {
        "fadd" | "zig" => 2 * n + 1,
        _ => n + 1,
    }
}

/// B3: `simplify` of a right-nested sum, a tower and a `sin` chain, on each thread of the bar.
/// HEAD: 3 248 for each on 2 MB debug; 32 640 on 8 MB release and 8 169 on 2 MB release. The
/// results keep the input's depth (HEAD's simplify combines no like terms of a right-nested sum).
#[test]
fn b3_simplify_depth_not_below_head() {
    let cases: &[(usize, usize)] = if DEBUG {
        &[(2 * MB, 3_120)]
    } else {
        &[(8 * MB, 31_300), (2 * MB, 7_840)]
    };
    for &(stack, n) in cases {
        for kind in ["rsum", "pow", "sin"] {
            let f = Arc::new(deep(kind, n));
            let g = f.clone();
            let d = on_stack(stack, move || tree_depth(&g.simplify()));
            assert_eq!(
                d,
                n + 1,
                "simplify of {kind} of depth {n} on a {} MB thread",
                stack / MB
            );
        }
    }
}

/// B3: `Expr::limit` (HEAD's strategies, which substitute and simplify) of a tower `x^(x^(...))`
/// at 1: `Finite(1)`. HEAD: depth 2 608 on 2 MB debug; 29 056 on 8 MB release, 7 261 on 2 MB
/// release.
#[test]
fn b3_limit_of_a_tower_depth_not_below_head() {
    let cases: &[(usize, usize)] = if DEBUG {
        &[(2 * MB, 2_500)]
    } else {
        &[(8 * MB, 27_900), (2 * MB, 6_970)]
    };
    for &(stack, n) in cases {
        let f = Arc::new(deep("pow", n));
        let g = f.clone();
        let r = on_stack(stack, move || {
            g.limit(&x(), &Expr::from(1), Direction::Both)
        });
        assert_eq!(
            r,
            LimitResult::Finite(Expr::from(1)),
            "limit of a tower of depth {n} at 1 on a {} MB thread",
            stack / MB
        );
    }
}

// ------------------------------------------------------------------------------------------------
// Review 2, B1: simplify (and Expr::limit) on function chains and zigzags, at least as deep as HEAD
// ------------------------------------------------------------------------------------------------

/// Review 2 (resources lens), B1: the fix of round 1 kept `simplify`'s own frame at HEAD's size but
/// put a frame of `simplify_function` (a call) or of `simplify_left_spine` (a binary node with a
/// binary left child) between two `simplify` frames, so these shapes recursed deeper per level than
/// HEAD. HEAD (8a0b310 built by cargo, review 2's `consume2.rs`, exact binary searches): release
/// 8 MB `fch` 14 553, `fadd` 10 075, `zig` 16 372; release 2 MB 3 631, 2 513, 4 084; debug 2 MB
/// `fch` 1 280, `fadd` 917, `zig` 1 628. The port of round 1 aborted from 12 779, 9 192 and 13 788
/// (release 8 MB), 3 189, 2 294 and 3 440 (release 2 MB) and 1 177 and 875 (debug 2 MB). The rows are
/// review 2's, at about 96 % of HEAD. The results keep the input's depth (HEAD's simplify folds no
/// like terms and leaves f(...) alone).
#[test]
fn b1r2_simplify_of_calls_and_zigzags_depth_not_below_head() {
    let cases: &[(usize, &'static str, usize)] = if DEBUG {
        &[
            (2 * MB, "fch", 1_228),
            (2 * MB, "fadd", 880),
            (2 * MB, "zig", 1_560),
        ]
    } else {
        &[
            (8 * MB, "fch", 13_970),
            (8 * MB, "fadd", 9_670),
            (8 * MB, "zig", 15_700),
            (2 * MB, "fch", 3_485),
            (2 * MB, "fadd", 2_410),
            (2 * MB, "zig", 3_920),
        ]
    };
    for &(stack, kind, n) in cases {
        let f = Arc::new(deep(kind, n));
        let g = f.clone();
        let d = on_stack(stack, move || tree_depth(&g.simplify()));
        assert_eq!(
            d,
            deep_depth(kind, n),
            "simplify of {kind} with {n} levels on a {} MB thread",
            stack / MB
        );
    }
}

/// Review 2, B1: `Expr::limit` at 1 (HEAD's code: substitute, then simplify) of `f(f(...f(x)))`,
/// which is `Unknown` (HEAD's limit laws have no rule for an unknown function), and of the zigzag
/// `L = (x + L) + x`, which is `Finite(2 n + 1)` (every x is 1). HEAD, release 8 MB: `fch` 13 433,
/// `zig` 14 553; the port of round 1 aborted from 12 779 and 13 786. Debug 2 MB: HEAD 1 125 and
/// 1 305 (the port of round 1 reached the same). At about 96 % of HEAD.
#[test]
fn b1r2_limit_of_calls_and_zigzags_depth_not_below_head() {
    let cases: &[(usize, &'static str, usize)] = if DEBUG {
        &[(2 * MB, "fch", 1_080), (2 * MB, "zig", 1_250)]
    } else {
        &[(8 * MB, "fch", 12_890), (8 * MB, "zig", 13_970)]
    };
    for &(stack, kind, n) in cases {
        let f = Arc::new(deep(kind, n));
        let g = f.clone();
        let r = on_stack(stack, move || {
            g.limit(&x(), &Expr::from(1), Direction::Both)
        });
        let want = if kind == "zig" {
            LimitResult::Finite(Expr::from(2 * n as i64 + 1))
        } else {
            LimitResult::Unknown
        };
        assert_eq!(
            r,
            want,
            "limit of {kind} with {n} levels at 1 on a {} MB thread",
            stack / MB
        );
    }
}

// ------------------------------------------------------------------------------------------------
// B2: limit_with_error is bounded by its budget (the limit through the engine, design §2.13)
// ------------------------------------------------------------------------------------------------

/// `limit_with_error` of the text `f` at `a`, order n, on a thread of `stack` bytes, timed.
fn lwe_text(f: String, a: &'static str, n: usize, stack: usize) -> (LimitResult, Option<i64>, f64) {
    on_stack(stack, move || {
        let e = parse(&f);
        let t0 = Instant::now();
        let (l, b) = lwe(&e, &x(), &parse(a), n);
        let secs = t0.elapsed().as_secs_f64();
        (l, b.as_ref().and_then(big_o_exponent), secs)
    })
}

/// B2: `x + x + ... + x` (n terms) at 1, order 3: `Finite(n)` with O(h^4) (n x = n + n h). HEAD's
/// `limit_with_error` overflowed from 3 000 terms (5 999 bytes) on a 2 MB debug thread and from
/// 30 000 terms (59 999 bytes) on an 8 MB release thread, in `Expr::limit` -> `substitute`.
#[test]
fn b2_limit_with_error_of_a_long_sum() {
    let (stack, n) = if DEBUG {
        (2 * MB, 3_000)
    } else {
        (8 * MB, 30_000)
    };
    let f = vec!["x"; n].join("+");
    let (l, b, secs) = lwe_text(f, "1", 3, stack);
    assert_eq!(l, LimitResult::Finite(Expr::from(n as i64)), "{n} terms");
    assert_eq!(b, Some(4), "{n} terms");
    assert!(secs < BAR_SECONDS, "{n} terms: {secs:.1} s");
}

/// The tree the parser built for `x^-x^-...^-x` with n carets: `x^(-(x^(-(...x^(-x)))))`, built
/// from the inside (no recursion). S1 finish: the parser now refuses the text from 1 017 levels
/// (`depth`, as HEAD refuses every such text), so the tests of `limit_with_error` on deep towers
/// build the same tree directly.
fn signed_tower(n: usize) -> Expr {
    let mut e = Expr::Symbol(x());
    for _ in 0..n {
        e = Expr::Binary(
            BinaryOp::Pow,
            Arc::new(Expr::Symbol(x())),
            Arc::new(Expr::Unary(UnaryOp::Neg, Arc::new(e))),
        );
    }
    e
}

/// `limit_with_error` of `f` at `a`, order n, on a thread of `stack` bytes, timed.
fn lwe_expr(f: Expr, a: &'static str, n: usize, stack: usize) -> (LimitResult, Option<i64>, f64) {
    on_stack(stack, move || {
        let t0 = Instant::now();
        let (l, b) = lwe(&f, &x(), &parse(a), n);
        let secs = t0.elapsed().as_secs_f64();
        (l, b.as_ref().and_then(big_o_exponent), secs)
    })
}

/// B2: `x^-x^-...^-x` (n levels) at 1, order 3. The value at 1 is 1 (every level is 1^(-1)). HEAD's
/// parser rejects the input (a sign after `^`); the port's `limit_with_error` overflowed from
/// 1 400 levels (4 201 bytes) on a 2 MB debug thread and from 13 000 levels (39 001 bytes) on an
/// 8 MB release thread, in `Expr::limit` -> `substitute`. Now: up to a few thousand levels the
/// order-5 expansion of the limit fits in `LIMIT_SERIES_WORK`, so `Finite(1)` with O(h^4) (the
/// expansion reaches order 3); at 13 000 levels it does not (`BudgetExceeded` at 1e8 units,
/// checked here through `series_expansion`), so `(Unknown, None)`, in a fraction of a second.
/// S1 finish: the text of more than 1 016 levels is a `ParseError` now (`p6_*`), so the tower is
/// built as the tree the parser made of it (`signed_tower`, checked against the parser at 500
/// levels); the assertions are unchanged.
#[test]
fn b2_limit_with_error_of_a_signed_tower() {
    let small = vec!["x"; 501].join("^-");
    assert!(
        on_stack(2 * MB, move || parse(&small) == signed_tower(500)),
        "signed_tower(500) is the parser's tree"
    );
    let cases: &[(usize, usize, bool)] = if DEBUG {
        &[(2 * MB, 1_400, true)]
    } else {
        &[(8 * MB, 3_000, true), (8 * MB, 13_000, false)]
    };
    for &(stack, n, finite) in cases {
        if !finite {
            let r = on_stack(stack, move || {
                se(
                    &signed_tower(n),
                    &x(),
                    &ExpansionPoint::At(parse("1")),
                    5,
                    Some(rustmath_symbolic::series::LIMIT_SERIES_WORK),
                )
                .err()
            });
            assert!(
                matches!(r, Some(SeriesError::BudgetExceeded { .. })),
                "{n} levels: {r:?}"
            );
        }
        let (l, b, secs) = lwe_expr(signed_tower(n), "1", 3, stack);
        if finite {
            assert_eq!(
                (&l, b),
                (&LimitResult::Finite(Expr::from(1)), Some(4)),
                "{n} levels"
            );
        } else {
            assert_eq!((&l, b), (&LimitResult::Unknown, None), "{n} levels");
        }
        assert!(secs < BAR_SECONDS, "{n} levels: {secs:.1} s");
    }
}

/// B2: `x + sin(1) + ... + sin(m)` at 0, order 2: the constant term is not rational, so the limit
/// path gives `Unknown` (design §2.13) and no `BigO`. HEAD's `limit_with_error` used 797 MB in
/// 1.81 s for m = 2 399 (22 884 bytes) in release, 1.41 GB for m = 3 199, and failed to allocate
/// under a 4 GB cap for m = 9 999 (98 884 bytes). Release: each within the bar's 20 s, and VmHWM
/// of this process under the bar's 512 MB after them (VmHWM covers every test of this binary run
/// so far, so the check is stricter than the bar's per-input limit).
#[test]
fn b2_limit_with_error_of_a_sum_of_kernels() {
    let ms: &[usize] = if DEBUG {
        &[2_399]
    } else {
        &[2_399, 3_199, 9_999]
    };
    let stack = if DEBUG { 2 * MB } else { 8 * MB };
    for &m in ms {
        let f = std::iter::once("x".to_string())
            .chain((1..=m).map(|k| format!("sin({k})")))
            .collect::<Vec<_>>()
            .join("+");
        let (l, b, secs) = lwe_text(f, "0", 2, stack);
        assert_eq!((&l, b), (&LimitResult::Unknown, None), "m = {m}");
        assert!(secs < BAR_SECONDS, "m = {m}: {secs:.1} s");
        if !DEBUG {
            if let Some(bytes) = vm_hwm_bytes() {
                assert!(bytes < BAR_BYTES, "m = {m}: VmHWM {} MB", bytes >> 20);
            }
        }
    }
}
