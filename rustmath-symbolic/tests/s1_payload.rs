//! S1 implementation, fix-up round 2: review 2's B2 (resources lens,
//! `fu2/s1impl/S1.review2-resources.md`), written as regression tests before the fix.
//!
//! B2: the result-tree cap (`MAX_RESULT_TREE`, red team round 3 structure H2, one of the owner's
//! three holes) counts nodes, not what they hold. A coefficient that is one shared kernel with a
//! large leaf (a long integer literal, a long symbol or function name) is rendered once and named by
//! every term, so the result is a small DAG whose tree repeats the leaf at every term. `simplify`
//! and `substitute` copy the leaf at each occurrence, and `Display` prints it each time. Review 2
//! measured, in release on an 8 MB thread, after `taylor_with_budget` with the REPL budget at
//! order 20 000 (`Ok` in about 0.02 s and 12 MB):
//!
//! | input (bytes), at 0 | `simplify` | `substitute` | `Display` |
//! |---|---|---|---|
//! | `sqrt(N)/(1-x)`, N of 99 950 digits (99 962) | 818 MB | 818 MB | 131 s, 1.9 GB |
//! | `sin(N)/(1-x)` (99 961) | 817 MB | | |
//! | a symbol of 99 941 characters over `(1-x)` (99 947) | 1.93 GB | 1.93 GB | |
//! | a function name of 99 931 characters, `f..(1)/(1-x)` (99 940) | 1.93 GB | 1.93 GB | |
//! | `1/(1+N-x)` at the point N, N of 49 990 digits (49 998 and 49 990) | 417 MB | 416 MB | |
//!
//! S1_BAR (c): an input of at most 100 KB through `taylor_with_budget` with the REPL's budget whose
//! consumers (`simplify`, `substitute`, `Display`) run longer than 20 s or use more than 512 MB is a
//! budget hole. The tests here require, for each input, a refusal or a result whose tree holds a
//! bounded amount of leaf text (both profiles, `b2r2_*`), and in release (`#[ignore]`) that each
//! consumer of an accepted result stays within the bar. The same inputs at order 200 must stay
//! values (no scope cut: their trees hold about 20 MB of leaf text).
//!
//! Run: `cargo test -p rustmath-symbolic --test s1_payload`, and in release
//! `cargo test --release -p rustmath-symbolic --test s1_payload -- --ignored --test-threads=1`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: an expansion of 1/(1-x) has every coefficient 1, here times the constant); S1
//! design §2.15 (structural caps) and S1_BAR (c).

mod s1_common;

use s1_common::api::*;
use s1_common::check::*;
use std::collections::HashMap;
use std::time::Instant;

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

/// The REPL's largest order (`rustmath-jupyter` `TAYLOR_ORDER`).
const TAYLOR_ORDER: usize = 20_000;

/// The structural cap of the design (§2.15, revision 7rt3): `MAX_RESULT_TREE` = 2^21 nodes.
const MAX_RESULT_TREE: u64 = 1 << 21;

/// The leaf text a result's tree may hold, from the bar: `Display` of a 10^5-digit integer prints
/// about 15 MB/s in release (review 2: 2.0 GB of text in 131 s), so 128 MiB of leaf text prints in
/// about 9 s, and `simplify` or `substitute` copies at most 128 MiB of leaves (an integer's copy is
/// smaller than its text). A result's tree also has at most `MAX_RESULT_TREE` nodes.
const LEAF_TEXT_BOUND: u64 = 128 << 20;

/// A `digits`-digit integer: a fixed pseudo-random digit string (a linear congruential generator),
/// first digit nonzero, last digit 7 (so not a perfect square).
fn big_n(digits: usize) -> String {
    let mut s = String::with_capacity(digits);
    let mut v: u64 = 0x2545_f491_4f6c_dd1d;
    for i in 0..digits {
        v = v
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mut d = ((v >> 33) % 10) as u8;
        if i == 0 && d == 0 {
            d = 1;
        }
        if i + 1 == digits {
            d = 7;
        }
        s.push((b'0' + d) as char);
    }
    s
}

/// Review 2's five inputs (text, point), each at most 100 KB.
fn inputs() -> Vec<(&'static str, String, String)> {
    let n = big_n(99_950);
    let m = big_n(49_990);
    vec![
        ("sqrt(N)/(1-x)", format!("sqrt({n})/(1-x)"), "0".to_string()),
        ("sin(N)/(1-x)", format!("sin({n})/(1-x)"), "0".to_string()),
        (
            "a long symbol over (1-x)",
            format!("{}/(1-x)", "a".repeat(99_941)),
            "0".to_string(),
        ),
        (
            "a long function name, f..(1)/(1-x)",
            format!("f{}(1)/(1-x)", "g".repeat(99_930)),
            "0".to_string(),
        ),
        ("1/(1+N-x) at N", format!("1/(1+{m}-x)"), m.clone()),
    ]
}

/// The characters of leaf text in e's tree: at each occurrence of a leaf, the length of its
/// `Display` (an integer's digits, both parts of a rational, a float, a symbol's name), and at each
/// call, its function name. Computed over the DAG (memoized by address, explicit stack), saturating.
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

/// B2: each input at the REPL's largest order through `taylor_with_budget` with the REPL budget,
/// on the bar's stack. Before the fix every one was `Ok` with a tree of 1 to 2 GB of leaf text
/// (20 001 terms, each naming the large leaf). Now each must be refused (a structural limit or the
/// budget), or be a result whose tree holds at most `LEAF_TEXT_BOUND` characters of leaf text and at
/// most `MAX_RESULT_TREE` nodes. The same holds through `series_expansion` (its coefficients and
/// its `to_expr`).
#[test]
fn b2r2_large_leaves_at_taylor_order_are_bounded() {
    let mut failures: Vec<String> = Vec::new();
    for (what, f, a) in inputs() {
        let w = what.to_string();
        let msg = on_stack(STACK, move || -> Result<(), String> {
            let fe = parse(&f);
            let ae = parse(&a);
            let check = |r: Result<Expr, SeriesError>, via: &str| -> Result<(), String> {
                match r {
                    Err(e) => match kind(&e) {
                        Kind::LimitExceeded(_) | Kind::BudgetExceeded => Ok(()),
                        k => Err(format!("{w} via {via}: unexpected refusal {k:?}")),
                    },
                    Ok(r) => {
                        let text = tree_leaf_text(&r);
                        let nodes = tree_size(&r);
                        if text > LEAF_TEXT_BOUND || nodes > MAX_RESULT_TREE {
                            Err(format!(
                                "{w} via {via}: Ok with {text} characters of leaf text and {nodes} nodes in its tree"
                            ))
                        } else {
                            Ok(())
                        }
                    }
                }
            };
            let a = check(
                tw(&fe, &x(), &ae, TAYLOR_ORDER, REPL_TAYLOR_WORK),
                "taylor_with_budget",
            );
            let s = se(
                &fe,
                &x(),
                &ExpansionPoint::At(ae.clone()),
                TAYLOR_ORDER as i64,
                Some(REPL_TAYLOR_WORK),
            );
            let b = check(s.map(|s| s.to_expr()), "series_expansion");
            match (a, b) {
                (Ok(()), Ok(())) => Ok(()),
                (a, b) => Err(format!(
                    "{}; {}",
                    a.err().unwrap_or_default(),
                    b.err().unwrap_or_default()
                )),
            }
        });
        if let Err(m) = msg {
            failures.push(m);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// B2, the other side: the same inputs at order 200 stay values (their trees hold about 10 to 20 MB
/// of leaf text). Each coefficient of 1/(1-h) is 1, so every coefficient of the result is the
/// constant of the input (sqrt(N), sin(N), the symbol, the call) or 1 (`1/(1+N-x)` at N, h = x - N):
/// 201 stored terms, of exponents 0 to 200, all equal to the first.
#[test]
fn b2r2_large_leaves_at_order_200_stay_values() {
    for (what, f, a) in inputs() {
        let w = what.to_string();
        let msg = on_stack(STACK, move || -> Result<(), String> {
            let s = se(
                &parse(&f),
                &x(),
                &ExpansionPoint::At(parse(&a)),
                200,
                Some(REPL_TAYLOR_WORK),
            )
            .map_err(|e| format!("{w} at order 200: expected Ok, got {e:?}"))?;
            if s.terms.len() != 201 {
                return Err(format!("{w}: {} stored terms", s.terms.len()));
            }
            let c0 = s.terms[0].1.clone();
            for (k, (e, c)) in s.terms.iter().enumerate() {
                if *e != k as i64 || **c != *c0 {
                    return Err(format!(
                        "{w}: term {k} has exponent {e} or another coefficient"
                    ));
                }
            }
            if what.starts_with("1/(1+N-x)") && *c0 != Expr::from(1) {
                return Err(format!("{w}: the coefficients are not 1"));
            }
            let r = tw(&parse(&f), &x(), &parse(&a), 200, REPL_TAYLOR_WORK)
                .map_err(|e| format!("{w} at order 200 via taylor_with_budget: {e:?}"))?;
            let text = tree_leaf_text(&r);
            if text > LEAF_TEXT_BOUND {
                return Err(format!("{w} at order 200: {text} characters of leaf text"));
            }
            Ok(())
        });
        if let Err(m) = msg {
            panic!("{m}");
        }
    }
}

/// `taylor_with_budget` of `f` at `a` to `order` with the REPL budget, then, for an `Ok` result,
/// `simplify`, `substitute(x, 2)`, `==` with a clone and `Display`, each timed, on one 8 MB thread
/// (the bar's release stack), with VmHWM of the process checked after each consumer (it covers
/// every call so far in this process, so the check is stricter than the bar's per-input 512 MB).
/// Returns whether the result was `Ok`.
fn consumers_within_bar(what: String, f: String, a: String, order: usize) -> bool {
    on_stack(8 << 20, move || {
        let hwm = |step: &str| {
            if let Some(b) = vm_hwm_bytes() {
                assert!(
                    b < BAR_BYTES,
                    "{what}, order {order}, {step}: VmHWM {} MB",
                    b >> 20
                );
            }
        };
        let fe = parse(&f);
        let t = Instant::now();
        let r = tw(&fe, &x(), &parse(&a), order, REPL_TAYLOR_WORK);
        let secs = t.elapsed().as_secs_f64();
        assert!(
            secs < BAR_SECONDS,
            "{what}, order {order}: taylor {secs:.1} s"
        );
        hwm("taylor_with_budget");
        let r = match r {
            Err(e) => {
                eprintln!("{what}, order {order}: refused ({e:?}) in {secs:.3} s");
                return false;
            }
            Ok(r) => r,
        };
        let step = |name: &str, g: &dyn Fn(&Expr) -> usize| {
            let t = Instant::now();
            let n = g(&r);
            let secs = t.elapsed().as_secs_f64();
            eprintln!("{what}, order {order}: {name} {secs:.3} s ({n})");
            assert!(
                secs < BAR_SECONDS,
                "{what}, order {order}: {name} {secs:.1} s"
            );
            hwm(name);
        };
        step("simplify", &|r| {
            let s = r.simplify();
            usize::from(s == *r)
        });
        step("substitute", &|r| {
            let s = r.substitute(&x(), &Expr::from(2));
            usize::from(matches!(s, Expr::Binary(..)))
        });
        step("==", &|r| usize::from(*r == r.clone()));
        step("Display", &|r| format!("{r}").len());
        let t = Instant::now();
        drop(r);
        drop(fe);
        let secs = t.elapsed().as_secs_f64();
        assert!(
            secs < BAR_SECONDS,
            "{what}, order {order}: drop {secs:.1} s"
        );
        true
    })
}

/// B2, the bar itself (release, 8 MB thread): each input at order 20 000, then each consumer of an
/// accepted result, within 20 s and 512 MB (VmHWM). Before the fix: `simplify` of `sqrt(N)/(1-x)`
/// took 818 MB and its `Display` 131 s. The same inputs at order 200 are `Ok`, and their consumers
/// stay within the bar too. Last, `sqrt(N)/(1-x)` at order 1300, just under the leaf-text cap of
/// the engine (2^27 characters: about 1.3e8 here), which prints the most digits of a 10^5-digit
/// integer that an accepted result can hold: a refusal, or every consumer within the bar.
/// `cargo test --release -p rustmath-symbolic --test s1_payload -- --ignored --test-threads=1`
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b2r2_consumers_of_large_leaf_results_release() {
    for (what, f, a) in inputs() {
        consumers_within_bar(what.to_string(), f.clone(), a.clone(), TAYLOR_ORDER);
        let ok = consumers_within_bar(what.to_string(), f, a, 200);
        assert!(ok, "{what} at order 200: expected Ok");
    }
    let (what, f, a) = inputs().swap_remove(0);
    consumers_within_bar(what.to_string(), f, a, 1300);
}
