//! S1 implementation, fix-up round 3: review 3's two blockers (resources lens,
//! `fu2/s1impl/S1.review3-resources.md`) and the minor items this round fixes (both lenses), as
//! regression tests written before the fixes.
//!
//! - B1: the leaf-text cap of a result (`MAX_RESULT_TEXT`, fix-up round 2) counted a float leaf as
//!   24 characters, but `Display` prints an f64 with `{}`, which never uses an exponent: 4.9e-324
//!   prints 326 characters, 1.0e-308 prints 310. `g(4.9e-324, ..., 11 000 arguments)/(1-x)`
//!   (99 008 bytes) at 0, order 189, was `Ok` through `taylor_with_budget` in 0.01 s, and the
//!   `Display` of the result was 685 MB of text (VmHWM 662 MB, release).
//! - B2: neither result cap bounded `simplify` of a result. HEAD's `simplify` folds c^n for every
//!   rational c and |n| <= 4096, with no bound on the size of c^n, and it folds a subtree again at
//!   every occurrence. A kernel that the engine made from an input node renders as that node, so
//!   `simplify` refolded the node at each term, and under a kernel power K^k it folded c^(4096 k).
//!   Release, 8 MB thread: `((2^256+1)^4096*x)^4096` (24 bytes) at order 4096 gives a 48-byte
//!   result whose `simplify` did not finish in 60 s; `1/(1-(2^256+1)^4096*x)` at order 100 took
//!   336 s and 730 MB; `exp(x+(2^256+1)^4096)` at order 5000 took 38 s and 667 MB;
//!   `sin((2^254+1)^4096)/(1-x)` at order 1000 took 7.7 s; `exp(x+N^4096)` with a 600-digit N
//!   took 95 s and 597 MB at order 600.
//! - Minors: the printing margin of large integers (review 3 resources m1: `Display` of
//!   `exp(x+(10^300-1)^1050*7^130000)` at order 312 took 18.8 s, 94 % of the bar);
//!   `display_estimate` counted a float as 24 (m2: the `UndecidedZero` message and
//!   `ExpansionPoint`'s `Debug` printed 14 times their limit); a function name in full in an error
//!   message (m3); `big_o()` at infinity was O(x^P) where the remainder is O(x^-P) (review 3
//!   semantics m1); a float literal with a nonzero digit that underflows parsed as the exact 0
//!   (semantics m2).
//!
//! S1_BAR (c): an input of at most 100 KB through `taylor_with_budget` with the REPL's budget whose
//! result's consumers run longer than 20 s or use more than 512 MB (release, 8 MB thread) is a
//! budget hole. The `#[ignore]` tests are the release measurements:
//! `cargo test --release -p rustmath-symbolic --test s1_review3 -- --ignored --test-threads=1`.
//! The other tests run in both profiles: `cargo test -p rustmath-symbolic --test s1_review3`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: the expansion of 1/(1-x) has every coefficient 1, here times a constant; §49.5:
//! exp and sin of a series); S1 design §2.9 (the public API, the O-term at infinity), §2.11
//! (Display), §2.12 (the literal parser), §2.15 (structural caps) and S1_BAR (c).

mod s1_common;

use rustmath_integers::Integer;
use rustmath_symbolic::BinaryOp;
use s1_common::api::*;
use s1_common::check::*;
use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

/// The structural cap of the design (§2.15, revision 7rt3): `MAX_RESULT_TREE` = 2^21 nodes.
const MAX_RESULT_TREE: u64 = 1 << 21;

/// The leaf text a result's tree may hold (`s1_payload.rs`, fix-up round 2): 128 MiB.
const LEAF_TEXT_BOUND: u64 = 128 << 20;

/// The printing margin of minor m1 (resources): an accepted result prints within half the bar.
const PRINT_MARGIN_SECONDS: f64 = BAR_SECONDS / 2.0;

/// `g(lit, lit, ..., lit)/(1-x)` with `n` arguments, no spaces (review 3's `in/gsub.txt` for
/// `4.9e-324` and 11 000 arguments: 99 008 bytes).
fn float_call(lit: &str, n: usize) -> String {
    format!("g({})/(1-x)", vec![lit; n].join(","))
}

/// The characters of leaf text in e's tree, as `Display` prints them: at each occurrence of a
/// leaf, the length of its `Display` (an integer's digits, both parts of a rational, a float, a
/// symbol's name), and at each call its function name. Over the DAG (memoized by address,
/// explicit stack), saturating. (The same count as `s1_payload.rs`.)
fn tree_leaf_text(root: &Expr) -> u64 {
    let mut memo: HashMap<*const Expr, u64> = HashMap::new();
    let mut stack: Vec<(&Expr, bool)> = vec![(root, false)];
    while let Some((n, done)) = stack.pop() {
        let key = n as *const Expr;
        if memo.contains_key(&key) {
            continue;
        }
        let children: Vec<&Expr> = match n {
            Expr::Binary(_, l, r) => vec![&**l, &**r],
            Expr::Unary(_, a) => vec![&**a],
            Expr::Function(_, args) => args.iter().map(|a| &**a).collect(),
            _ => vec![],
        };
        if !done && !children.is_empty() {
            stack.push((n, true));
            for c in children {
                stack.push((c, false));
            }
            continue;
        }
        let own = match n {
            Expr::Integer(_) | Expr::Rational(_) | Expr::Real(_) | Expr::Symbol(_) => {
                format!("{n}").len() as u64
            }
            Expr::Function(name, _) => name.len() as u64,
            _ => 0,
        };
        let s = children
            .iter()
            .fold(own, |acc, c| acc.saturating_add(memo[&(*c as *const Expr)]));
        memo.insert(key, s);
    }
    memo[&(root as *const Expr)]
}

/// The largest bit length of an integer, or of a numerator or denominator, in e (DAG walk).
fn max_literal_bits(root: &Expr) -> u64 {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack: Vec<&Expr> = vec![root];
    let mut best = 0u64;
    while let Some(n) = stack.pop() {
        if !seen.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Integer(i) => best = best.max(i.bit_length()),
            Expr::Rational(q) => {
                best = best
                    .max(q.numerator().bit_length())
                    .max(q.denominator().bit_length())
            }
            Expr::Binary(_, l, r) => {
                stack.push(&**l);
                stack.push(&**r);
            }
            Expr::Unary(_, a) => stack.push(&**a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    best
}

/// The distinct nodes (by address) of e that are integers of more than `bits` bits.
fn distinct_big_integers(root: &Expr, bits: u64) -> usize {
    let mut seen: HashSet<*const Expr> = HashSet::new();
    let mut stack: Vec<&Expr> = vec![root];
    let mut count = 0usize;
    while let Some(n) = stack.pop() {
        if !seen.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Integer(i) if i.bit_length() > bits => count += 1,
            Expr::Binary(_, l, r) => {
                stack.push(&**l);
                stack.push(&**r);
            }
            Expr::Unary(_, a) => stack.push(&**a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    count
}

/// Run `g(&r)` on its own thread with an 8 MB stack and wait at most `secs` seconds. `Some` with
/// the value and the wall time when it returns in time, `None` on a timeout. After a timeout the
/// thread keeps running until the test binary exits (the runs on the code before the fix used
/// `ulimit -v`), and the caller fails the test.
fn within<T: Send + 'static>(secs: f64, r: Arc<Expr>, g: fn(&Expr) -> T) -> Option<(T, f64)> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .stack_size(8 << 20)
        .spawn(move || {
            let t = Instant::now();
            let v = g(&r);
            let _ = tx.send((v, t.elapsed().as_secs_f64()));
        })
        .expect("spawn");
    rx.recv_timeout(Duration::from_secs_f64(secs)).ok()
}

// ---------------------------------------------------------------------------------------------
// B1: float leaves

/// B1 (both profiles): review 3's inputs, `g(4.9e-324 x 11 000)/(1-x)` and the same with
/// `1.0e-308`, at 0, order 189, through `taylor_with_budget` with the REPL budget and through
/// `series_expansion` (its `to_expr`). Before the fix both were `Ok` with about 6.8e8 characters
/// of leaf text (189 terms x 11 000 floats x 310 to 326 characters), counted by the cap as 5.0e7.
/// Now each must be refused (a structural limit or the budget), or be a result whose tree holds at
/// most `LEAF_TEXT_BOUND` characters of leaf text as `Display` prints them, and at most
/// `MAX_RESULT_TREE` nodes. Order 1 stays a value (no scope cut: 2 terms, 7 MB of text).
#[test]
fn b1r3_float_leaves_at_order_189_are_bounded() {
    let mut failures: Vec<String> = Vec::new();
    for lit in ["4.9e-324", "1.0e-308"] {
        let f = float_call(lit, 11_000);
        assert!(f.len() <= 100_000, "{} bytes", f.len());
        let lit = lit.to_string();
        let msg = on_stack(STACK, move || -> Result<(), String> {
            let fe = parse(&f);
            let check = |r: Result<Expr, SeriesError>, via: &str| -> Result<(), String> {
                match r {
                    Err(e) => match kind(&e) {
                        Kind::LimitExceeded(_) | Kind::BudgetExceeded => Ok(()),
                        k => Err(format!("{lit} via {via}: unexpected refusal {k:?}")),
                    },
                    Ok(r) => {
                        let text = tree_leaf_text(&r);
                        let nodes = tree_size(&r);
                        if text > LEAF_TEXT_BOUND || nodes > MAX_RESULT_TREE {
                            Err(format!(
                                "{lit} via {via}: Ok with {text} characters of leaf text and {nodes} nodes in its tree"
                            ))
                        } else {
                            Ok(())
                        }
                    }
                }
            };
            let a = check(
                tw(&fe, &x(), &Expr::from(0), 189, REPL_TAYLOR_WORK),
                "taylor_with_budget",
            );
            let s = se(
                &fe,
                &x(),
                &ExpansionPoint::At(Expr::from(0)),
                189,
                Some(REPL_TAYLOR_WORK),
            );
            let b = check(s.map(|s| s.to_expr()), "series_expansion");
            // order 1: two terms, each the shared call, so a value
            let one = tw(&fe, &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK)
                .map(|_| ())
                .map_err(|e| format!("{lit} at order 1: expected Ok, got {e:?}"));
            let errs: Vec<String> = [a, b, one].into_iter().filter_map(|r| r.err()).collect();
            if errs.is_empty() {
                Ok(())
            } else {
                Err(errs.join("; "))
            }
        });
        if let Err(m) = msg {
            failures.push(m);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// B1, the bar itself (release, 8 MB thread): `in/gsub.txt` at 0, order 189, through
/// `taylor_with_budget` with 1e8: refused, or `Display` of the result (into a `String`), and
/// `simplify`, `substitute` and `==`, each within 20 s with VmHWM under 512 MB. Before the fix:
/// `Display` 0.43 s, VmHWM 662 MB.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b1r3_display_of_float_results_release() {
    let f = float_call("4.9e-324", 11_000);
    let r = on_stack(8 << 20, move || {
        let t = Instant::now();
        let r = tw(&parse(&f), &x(), &Expr::from(0), 189, REPL_TAYLOR_WORK);
        let secs = t.elapsed().as_secs_f64();
        assert!(secs < BAR_SECONDS, "taylor {secs:.1} s");
        r
    });
    let r = match r {
        Err(e) => {
            eprintln!("gsub at order 189: refused ({e:?})");
            return;
        }
        Ok(r) => Arc::new(r),
    };
    consumers(
        "gsub at order 189",
        &r,
        BAR_SECONDS,
        &[
            ("Display", |r| format!("{r}").len()),
            ("simplify", |r| usize::from(r.simplify() == *r)),
            ("substitute", |r| {
                usize::from(matches!(
                    r.substitute(&x(), &Expr::from(2)),
                    Expr::Binary(..)
                ))
            }),
            ("==", |r| usize::from(*r == r.clone())),
        ],
    );
}

/// A named consumer of a result.
type Consumer<'a> = (&'a str, fn(&Expr) -> usize);

/// Each named consumer of `r` on its own 8 MB thread, within `secs` and VmHWM under the bar.
fn consumers(what: &str, r: &Arc<Expr>, secs: f64, steps: &[Consumer<'_>]) {
    for (name, g) in steps {
        match within(secs, r.clone(), *g) {
            None => panic!("{what}: {name} did not finish within {secs} s"),
            Some((n, t)) => {
                eprintln!("{what}: {name} {t:.3} s ({n})");
                assert!(t < secs, "{what}: {name} {t:.1} s");
            }
        }
        if let Some(b) = vm_hwm_bytes() {
            assert!(b < BAR_BYTES, "{what}: {name}: VmHWM {} MB", b >> 20);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// B2: simplify of a result

/// B2 (both profiles): a kernel power whose base the engine refused to compute (its a-priori size,
/// 4096 x 258 bits, passes the engine's 2^20-bit cap of an exact power) is rendered as the input
/// node. Before the fix `simplify` of the 48-byte result folded (2^256+1)^4096 (1.05e6 bits) and
/// then raised it to the 4096th power (4.3e9 bits): not done after 60 s in release. Now
/// `simplify` returns at once, and it builds no literal above the engine's cap (2^20 bits) either:
/// the result is unchanged in value, the power stays a power.
#[test]
fn b2r3_simplify_of_a_refused_kernel_power_returns() {
    let deadline = if cfg!(debug_assertions) { 60.0 } else { 20.0 };
    let r = on_stack(STACK, || {
        tw(
            &parse("((2^256+1)^4096*x)^4096"),
            &x(),
            &Expr::from(0),
            4096,
            REPL_TAYLOR_WORK,
        )
    })
    .expect("a value: the 4096th power of a kernel times x");
    assert!(tree_leaf_text(&r) < 100, "the result is a short monomial");
    let r = Arc::new(r);
    let (bits, secs) = within(deadline, r.clone(), |r| max_literal_bits(&r.simplify()))
        .unwrap_or_else(|| panic!("simplify of the result did not finish within {deadline} s"));
    assert!(
        bits <= 1 << 20,
        "simplify built a literal of {bits} bits in {secs:.1} s"
    );
}

/// B2 (both profiles): a kernel made from an input node renders as that node, so the node is
/// shared by every term of the result. `sin((2^200+1)^512)/(1-x)` at 0, order 60: 61 terms name
/// the kernel `sin((2^200+1)^512)`, whose source (2^200+1)^512 (1.0e5 bits) `simplify` folds.
/// Before the fix it folded the shared source once per term (61 distinct integers of 1.0e5 bits in
/// its output). Now each shared node is simplified once and its value is shared: one integer, the
/// same value at every term (each coefficient is sin((2^200+1)^512)).
#[test]
fn b2r3_simplify_folds_a_shared_source_once() {
    let (r, s) = on_stack(STACK, || {
        let r = tw(
            &parse("sin((2^200+1)^512)/(1-x)"),
            &x(),
            &Expr::from(0),
            60,
            REPL_TAYLOR_WORK,
        )
        .expect("a value");
        let s = r.simplify();
        (r, s)
    });
    let want = {
        let b = Integer::from(2).pow(200) + Integer::from(1);
        b.pow(512)
    };
    assert_eq!(
        distinct_big_integers(&r, 1 << 16),
        0,
        "the result holds the source text"
    );
    assert_eq!(
        distinct_big_integers(&s, 1 << 16),
        1,
        "simplify's output should hold the folded source once"
    );
    // every coefficient of simplify's output is sin((2^200+1)^512): the folded value
    let mut seen = 0usize;
    let mut stack: Vec<&Expr> = vec![&s];
    let mut visited: HashSet<*const Expr> = HashSet::new();
    while let Some(n) = stack.pop() {
        if !visited.insert(n as *const Expr) {
            continue;
        }
        match n {
            Expr::Integer(i) if i.bit_length() > 1 << 16 => {
                assert_eq!(*i, want);
                seen += 1;
            }
            Expr::Binary(_, l, r) => {
                stack.push(&**l);
                stack.push(&**r);
            }
            Expr::Unary(_, a) => stack.push(&**a),
            Expr::Function(_, args) => stack.extend(args.iter().map(|a| &**a)),
            _ => {}
        }
    }
    assert_eq!(seen, 1);
    // the value of each term is unchanged: simplify of one coefficient equals the shared node
    assert!(tree_size(&s) <= tree_size(&r));
}

/// B2, the bar itself (release, 8 MB thread): review 3's five inputs at the orders it measured,
/// through `taylor_with_budget` with 1e8. Each must be refused, or `simplify` of the result, and
/// `Display` of the result and of `simplify`'s output, must each finish within 20 s with VmHWM
/// under 512 MB. Before the fix (review 3, release): not done after 60 s; 36.7 s; 336.5 s and
/// 730 MB; 38.3 s and 667 MB; 7.7 s; 31.6 s; 95.4 s and 597 MB.
fn simplify_within_bar(what: &str, f: &str, order: usize) {
    let fs = f.to_string();
    let r = on_stack(8 << 20, move || {
        let t = Instant::now();
        let r = tw(&parse(&fs), &x(), &Expr::from(0), order, REPL_TAYLOR_WORK);
        let secs = t.elapsed().as_secs_f64();
        assert!(secs < BAR_SECONDS, "taylor {secs:.1} s");
        r
    });
    let what = format!("{what}, order {order}");
    let r = match r {
        Err(e) => {
            eprintln!("{what}: refused ({e:?})");
            return;
        }
        Ok(r) => Arc::new(r),
    };
    consumers(
        &what,
        &r,
        BAR_SECONDS,
        &[
            ("simplify", |r| usize::from(r.simplify() == *r)),
            ("Display of simplify", |r| format!("{}", r.simplify()).len()),
            ("Display", |r| format!("{r}").len()),
        ],
    );
}

#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r3_simplify_of_results_release_power_of_kernel() {
    simplify_within_bar("((2^256+1)^4096*x)^4096", "((2^256+1)^4096*x)^4096", 4096);
}

#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r3_simplify_of_results_release_geometric() {
    for order in [20, 40, 100] {
        simplify_within_bar(
            "1/(1 - (2^256+1)^4096*x)",
            "1/(1 - (2^256+1)^4096*x)",
            order,
        );
    }
}

#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r3_simplify_of_results_release_exp() {
    for order in [1000, 5000] {
        simplify_within_bar("exp(x + (2^256+1)^4096)", "exp(x + (2^256+1)^4096)", order);
    }
}

#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r3_simplify_of_results_release_sin() {
    for order in [1000, 5000] {
        simplify_within_bar(
            "sin((2^254+1)^4096)/(1-x)",
            "sin((2^254+1)^4096)/(1-x)",
            order,
        );
    }
}

#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r3_simplify_of_results_release_exp_of_a_power_of_n() {
    // review 3's in/expNpow.txt: N = 10^599 + 7 written out (600 digits), 615 bytes
    let n = format!("1{}7", "0".repeat(598));
    let f = format!("exp(x + {n}^4096)");
    for order in [200, 600] {
        simplify_within_bar("exp(x + N^4096)", &f, order);
    }
}

// ---------------------------------------------------------------------------------------------
// Minor m1 (resources): the printing margin of large integers

/// Minor m1 (release, 8 MB thread): an accepted result prints within half the bar. Review 3's
/// `exp(x + (10^300-1)^1050*7^130000)` (34 bytes) at order 312 holds, at each of its 313 terms, the
/// shared kernel exp(C) with C of 424 850 digits: 133 MB of text, just under the text cap, which
/// printed in 18.8 s (num-bigint's decimal conversion is superlinear). The same constant over
/// (1-x)^2 has the distinct coefficients (k+1) C. Each must be refused, or print within 10 s.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn m1r3_large_integers_print_with_a_margin_release() {
    for (f, order) in [
        ("exp(x + (10^300-1)^1050*7^130000)", 312usize),
        ("(10^300-1)^1050*7^130000/(1-x)^2", 312),
    ] {
        let fs = f.to_string();
        let r = on_stack(8 << 20, move || {
            tw(&parse(&fs), &x(), &Expr::from(0), order, REPL_TAYLOR_WORK)
        });
        let r = match r {
            Err(e) => {
                eprintln!("{f}, order {order}: refused ({e:?})");
                continue;
            }
            Ok(r) => Arc::new(r),
        };
        consumers(
            &format!("{f}, order {order}"),
            &r,
            PRINT_MARGIN_SECONDS,
            &[("Display", |r| format!("{r}").len())],
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Minor m2 (resources): display_estimate and floats

/// Minor m2: the `Debug` of a point and the message of `UndecidedZero` are bounded (60 and 200
/// characters plus their fixed words); a float counted as 24 characters let them print 326 per
/// float.
#[test]
fn m2r3_bounded_messages_count_floats_by_their_display() {
    let p = format!("{:?}", ExpansionPoint::At(Expr::Real(5e-324)));
    assert!(p.len() < 120, "Debug of the point: {} characters", p.len());
    let p = format!("{:?}", ExpansionPoint::At(Expr::Real(0.5)));
    assert_eq!(p, "At(0.5)");
    // 1.0e-308*(0.5 - 1/2) is undecided (a float is an opaque kernel, design §2.2: no float
    // folding), and its printed form holds 1.0e-308 (310 characters) twice
    let r = on_stack(STACK, || {
        tw(
            &parse("1/(x + 1.0e-308*(0.5 - 1/2))"),
            &x(),
            &Expr::from(0),
            2,
            REPL_TAYLOR_WORK,
        )
    });
    match r {
        Err(e @ SeriesError::UndecidedZero(_)) => {
            let m = format!("{e}");
            assert!(
                m.len() < 400,
                "UndecidedZero message: {} characters",
                m.len()
            );
        }
        other => panic!("expected UndecidedZero, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Minor m3 (resources): names in error messages

/// Minor m3: an error message names an input function by at most a bounded prefix of its name.
/// `g...g(x)` with a 99 990-character name at 0 is `Unsupported` (an unknown function of the
/// variable); before the fix the message held the whole name.
#[test]
fn m3r3_error_messages_bound_function_names() {
    let name = "g".repeat(99_990);
    let f = format!("{name}(x)");
    let r = on_stack(STACK, move || {
        tw(&parse(&f), &x(), &Expr::from(0), 3, REPL_TAYLOR_WORK)
    });
    match r {
        Err(SeriesError::Unsupported(m)) => {
            assert!(m.len() < 300, "message of {} characters", m.len());
            assert!(m.starts_with("ggg"), "{m}");
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Semantics minors

/// Review 3 semantics m1: at infinity h = 1/var, so `big_o()` is O((1/x)^precision).
/// gp: `subst(x/(x+1), x, 1/y) + O(y^5)` = 1 - y + y^2 - y^3 + y^4 + O(y^5): the remainder of the
/// expansion to the exponent 4 is O(x^-5). Before the fix `big_o()` was O(x^5).
#[test]
fn s1r3_big_o_at_infinity_is_a_power_of_one_over_x() {
    let f = parse("x/(x+1)");
    let s = se(&f, &x(), &ExpansionPoint::Infinity, 4, None).expect("a value");
    assert_eq!(s.precision, 5);
    let b = s.big_o();
    // O((1/x)^5): at x = 2 the function is 1/32
    let at2 = b.function.substitute(&x(), &Expr::from(2)).simplify();
    assert_eq!(at2, parse("1/32"), "big_o() at infinity is {}", b.function);
    // at a finite point it is still O(h^precision)
    let s = se(&f, &x(), &ExpansionPoint::At(Expr::from(0)), 4, None).expect("a value");
    let at2 = s
        .big_o()
        .function
        .substitute(&x(), &Expr::from(2))
        .simplify();
    assert_eq!(at2, Expr::from(32));
    // and the terms at infinity, in h = 1/x: 1 - h + h^2 - h^3 + h^4
    let s = se(&f, &x(), &ExpansionPoint::Infinity, 4, None).expect("a value");
    let c: Vec<Expr> = (0..=4)
        .map(|k| (*s.coefficient(k).expect("below the precision")).clone())
        .collect();
    assert_eq!(
        c,
        vec![
            Expr::from(1),
            Expr::from(-1),
            Expr::from(1),
            Expr::from(-1),
            Expr::from(1)
        ]
    );
}

/// Review 3 semantics m2: a float literal with a nonzero digit whose f64 value is 0 (it
/// underflows) is refused by the parser, as the engine refuses a literal that overflows to inf.
/// Before the fix `1.0e-400` parsed as the exact 0, so `x/1.0e-400` at 0 was
/// `DivisionByZero("division by the constant 0")` and `1/(x + 1.0e-400)` was 1/x. A literal whose
/// digits are all 0 is the float 0, and a subnormal literal is its exact binary value.
#[test]
fn s1r3_float_literal_underflowing_to_zero_is_refused() {
    for s in [
        "1.0e-400",
        "x/1.0e-400",
        "1/(x + 1.0e-400)",
        "atanh(1 + 1.0e-400)*x",
        "2.5E-999",
        "-1.0e-400",
        "x^1.0e-400",
        "sin(1.0e-400)",
        "f(x, 1.0e-400)",
        "(1.0e-400)",
        "2*1.0e-400",
    ] {
        assert!(
            rustmath_symbolic::parse(s).is_err(),
            "{s} parsed as {:?}",
            rustmath_symbolic::parse(s)
        );
    }
    assert_eq!(parse("0.0e-400"), Expr::Real(0.0));
    assert_eq!(parse("0.000"), Expr::Real(0.0));
    assert_eq!(parse("4.9e-324"), Expr::Real(5e-324));
    assert_eq!(parse("1.0e-300"), Expr::Real(1e-300));
    // the refusal names the literal
    for s in ["x + 1.0e-400", "x^1.0e-400", "f(x, 1.0e-400)"] {
        let m = format!("{}", rustmath_symbolic::parse(s).unwrap_err());
        assert!(
            m.contains("1.0e-400") && m.contains("underflows"),
            "{s}: {m}"
        );
    }
    // unchanged: a float literal as a whole, and a quotient
    assert_eq!(
        parse("1.5/2"),
        Expr::Binary(
            BinaryOp::Div,
            Arc::new(Expr::Real(1.5)),
            Arc::new(Expr::from(2))
        )
    );
}
