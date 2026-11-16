//! S1 implementation, fix-up round 5: review 5's blockers that are not about the peak memory
//! (`fu2/s1impl/S1.review5-semantics.md` B1, `fu2/s1impl/S1.review5-resources.md` B1), as
//! regression tests written before the fixes. Blocker B2 of the resources lens (the memory of
//! `simplify`'s memo) is in `s1_review5_memory.rs`, a test binary of its own, because it measures
//! the peak memory of the process.
//!
//! - Semantics B1: the engine accepts any symbol as the variable. With the variable `pi`, the
//!   input's own `pi` is the variable, but the engine also made the constant pi itself, as the
//!   kernel of the symbol `pi`: the exact values of fix-up round 4 (arcsin(+-1) = +-pi/2,
//!   arccos(-1) = pi, arccos(0) = pi/2, arctan(+-1) = +-pi/4, arcsinh(+-i) = +-i pi/2) and erf's
//!   2/sqrt(pi). With the variable `I` (the parser maps `i` to `I`), the same held for the constant
//!   i of arccosh(-1) = i pi and arccosh(0) = i pi/2. Inside the engine a kernel and the variable
//!   are distinct, but the rendering writes both as the same symbol, so the returned expression is
//!   another function of the variable: `taylor(acos(0*pi), pi, 0, 2)` was `(1/2)*pi`, where the
//!   answer is the constant pi/2 (HEAD gives `arccos(0)`), and `subs(t, pi, 0)` gave 0.
//! - Resources B1: the fold cap of round 4 counted the work of `simplify`'s folds once per distinct
//!   node of the result's DAG. `substitute`, `expand` and `differentiate` rebuild the tree, so the
//!   next `simplify` folds a kernel's source once per term. `sin(K/3 x 2535)/(1-x)` (5 095 bytes)
//!   at 0, order 10, release: `taylor_with_budget` `Ok` in 0.016 s, then
//!   `simplify(R.substitute(x, 2))` 24.1 s, `simplify(R.expand())` 23.7 s and
//!   `simplify(R.differentiate(x))` 27.1 s; at order 190, 418.6 s.
//!
//! S1_BAR (a), a wrong result, (b), a regression against HEAD on an input HEAD handles, and (c),
//! an input of at most 100 KB whose result's consumers pass 20 s or 512 MB. The `#[ignore]` tests
//! are the release measurements:
//! `cargo test --release -p rustmath-symbolic --test s1_review5 -- --ignored --test-threads=1`.
//! The other tests run in both profiles: `cargo test -p rustmath-symbolic --test s1_review5`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: the expansion of 1/(1-x) has every coefficient 1, here times a constant; §49.5:
//! the transcendental functions of a series and their inverses); S1 design §2.2 (the constant
//! field), §2.9 (the public API), §2.10 (`simplify`), §2.15 (structural caps). Reference values:
//! PARI/GP 2.15.4 (`/usr/bin/gp`), `\p 70`: `acos(0)`, `atan(1)`, `acos(-1)`, `acosh(0)`,
//! `acosh(-1)`, `asinh(I)`, `2/sqrt(Pi)`, `-2/(3*sqrt(Pi))`, `Pi/6`, `Pi/3` (the script is
//! `fu2/s1impl/fix5/refs1.gp`, run as `gp -q < refs1.gp`).

mod s1_common;

use s1_common::api::*;
use s1_common::check::*;
use s1_common::eval;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

/// S1_BAR (c): 20 s and 512 MB.
const BAR_SECONDS: f64 = 20.0;
const BAR_BYTES: u64 = 512 << 20;

/// K = (2^254+1)^4096: exactly 2^20 bits a priori, the engine's and `simplify`'s largest exact power.
const K: &str = "(2^254+1)^4096";

// gp, `\p 70`
const ACOS_0: &str = "1.570796326794896619231321691639751442098584699687552910487472296153908";
const ATAN_1: &str = "0.7853981633974483096156608458198757210492923498437764552437361480769541";
const PI: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816";
const PI_6: &str = "0.5235987755982988730771072305465838140328615665625176368291574320513027";
const PI_3: &str = "1.047197551196597746154214461093167628065723133125035273658314864102605";
const TWO_OVER_SQRT_PI: &str =
    "1.128379167095512573896158903121545171688101258657997713688171443421285";
const MINUS_TWO_THIRDS_OVER_SQRT_PI: &str =
    "-0.3761263890318375246320529677071817238960337528859992378960571478070950";

// ---------------------------------------------------------------------------------------------
// Semantics B1: a variable named pi or I

/// A row: input, the variable's name, the point, the order, the coefficients (exponent, re, im;
/// absent ones are 0).
type VarRow = (
    &'static str,
    &'static str,
    &'static str,
    i64,
    &'static [Coef<'static>],
);

const VAR_ROWS: &[VarRow] = &[
    // the variable pi: arccos(0) = pi/2, arctan(1) = pi/4, arccos(-1) = pi, arcsin(1) = pi/2,
    // arcsin(-1) = -pi/2, arctan(-1) = -pi/4, arcsinh(i) = i pi/2, erf'(0) = 2/sqrt(pi)
    ("acos(0*pi)", "pi", "0", 2, &[(0, ACOS_0, "0")]),
    ("acos(pi - pi)", "pi", "0", 2, &[(0, ACOS_0, "0")]),
    ("atan(pi^0)", "pi", "1", 2, &[(0, ATAN_1, "0")]),
    ("atan(pi/pi)", "pi", "1", 2, &[(0, ATAN_1, "0")]),
    (
        "pi*acos(pi*0)",
        "pi",
        "1",
        2,
        &[(0, ACOS_0, "0"), (1, ACOS_0, "0")],
    ),
    (
        "acos(0*pi) + sin(pi)",
        "pi",
        "0",
        3,
        &[(0, ACOS_0, "0"), (1, "1", "0"), (3, "-1/6", "0")],
    ),
    (
        "acos(-1) + pi",
        "pi",
        "0",
        2,
        &[(0, PI, "0"), (1, "1", "0")],
    ),
    (
        "asin(1)*pi",
        "pi",
        "1",
        2,
        &[(0, ACOS_0, "0"), (1, ACOS_0, "0")],
    ),
    (
        "asin(-pi^0)",
        "pi",
        "0",
        2,
        &[(
            0,
            "-1.570796326794896619231321691639751442098584699687552910487472296153908",
            "0",
        )],
    ),
    (
        "atan(-pi^0)",
        "pi",
        "0",
        2,
        &[(
            0,
            "-0.7853981633974483096156608458198757210492923498437764552437361480769541",
            "0",
        )],
    ),
    ("asinh(I*pi^0)", "pi", "0", 2, &[(0, "0", ACOS_0)]),
    (
        "erf(pi)",
        "pi",
        "0",
        3,
        &[
            (1, TWO_OVER_SQRT_PI, "0"),
            (3, MINUS_TWO_THIRDS_OVER_SQRT_PI, "0"),
        ],
    ),
    // fix-up round 5's exact values (written after that fix, with the same check): arcsin(1/2) =
    // pi/6, arcsinh(i/2) = i pi/6, arctanh(i) = i pi/4
    ("asin(pi^0/2)", "pi", "0", 2, &[(0, PI_6, "0")]),
    ("asinh(I*pi^0/2)", "pi", "0", 2, &[(0, "0", PI_6)]),
    ("atanh(I*pi^0)", "pi", "0", 2, &[(0, "0", ATAN_1)]),
    ("acosh(I^0/2)", "I", "0", 2, &[(0, "0", PI_3)]),
    // the variable I (the parser maps i to I): arccosh(-1) = i pi, arccosh(0) = i pi/2
    ("acosh(-I^0)", "I", "0", 2, &[(0, "0", PI)]),
    ("acosh(0*I)", "I", "0", 2, &[(0, "0", ACOS_0)]),
    ("acosh(0*i)", "I", "0", 2, &[(0, "0", ACOS_0)]),
    ("acosh(-1) + I", "I", "0", 2, &[(0, "0", PI), (1, "1", "0")]),
    (
        "acosh(0*I) + sin(I)",
        "I",
        "0",
        3,
        &[(0, "0", ACOS_0), (1, "1", "0"), (3, "-1/6", "0")],
    ),
];

/// One row: through `series_expansion` (REPL budget) every coefficient is free of the variable's
/// name and equals gp; through `taylor_with_budget` the result `r` gives f(a) at the point:
/// `r.substitute(var, a)`, simplified and evaluated, is the coefficient of h^0. A typed refusal
/// (the same kind on both paths) is also accepted: it is honest.
fn check_var_row(row: &VarRow) -> Result<(), String> {
    let (f_text, name, a_text, order, want) = *row;
    let f = parse(f_text);
    let a = parse(a_text);
    let var = Symbol::new(name);
    let s = se(
        &f,
        &var,
        &ExpansionPoint::At(a.clone()),
        order,
        Some(REPL_TAYLOR_WORK),
    );
    let t = tw(&f, &var, &a, order as usize, REPL_TAYLOR_WORK);
    let s = match (s, t) {
        (Err(e1), Err(e2)) => {
            if kind(&e1) != kind(&e2) {
                return Err(format!("refusals differ: {e1:?} and {e2:?}"));
            }
            eprintln!("{f_text} in {name} at {a_text}: refused ({e1:?})");
            return Ok(());
        }
        (Ok(s), Ok(t)) => {
            if t != s.to_expr() {
                return Err("taylor_with_budget differs from to_expr()".into());
            }
            (s, t)
        }
        (s, t) => return Err(format!("the two paths disagree: {s:?} and {t:?}")),
    };
    let (s, t) = s;
    for (k, c) in &s.terms {
        if mentions(c, name) {
            return Err(format!(
                "the coefficient of h^{k} names the variable {name}: {c}"
            ));
        }
    }
    let lo = s.first_exponent().unwrap_or(0).min(0);
    for k in lo..=order {
        let got = match s.coefficient(k) {
            Some(c) => eval::eval(&c).map_err(|e| format!("coefficient {k} = {c}: {e}"))?,
            None => return Err(format!("no coefficient {k}")),
        };
        let w = want
            .iter()
            .find(|w| w.0 == k)
            .map(|(_, re, im)| eval::parse_coefficient(re, im))
            .unwrap_or_else(eval::zero);
        if !eval::close(&got, &w, 50) {
            return Err(format!(
                "coefficient of h^{k}: engine {} (as {:?}), gp {}",
                eval::show(&got),
                s.coefficient(k).map(|c| c.to_string()),
                eval::show(&w)
            ));
        }
    }
    // f(a): the result substituted at the point
    let at = t.substitute(&var, &a).simplify();
    let got = eval::eval(&at).map_err(|e| format!("{t} at {name} = {a_text} is {at}: {e}"))?;
    let w = want
        .iter()
        .find(|w| w.0 == 0)
        .map(|(_, re, im)| eval::parse_coefficient(re, im))
        .unwrap_or_else(eval::zero);
    if !eval::close(&got, &w, 50) {
        return Err(format!(
            "the result {t} at {name} = {a_text} is {} ({at}), f(a) is {}",
            eval::show(&got),
            eval::show(&w)
        ));
    }
    Ok(())
}

/// Semantics B1 (both profiles): with the variable named `pi` or `I`, no result writes a constant
/// with the variable's name. Before the fix: `acos(0*pi)` in pi at 0 was `(1/2)*pi`,
/// `atan(pi^0)` at 1 was `(1/4)*pi`, `acos(-1) + pi` was `pi + pi`, `erf(pi)` was
/// `2*pi^(-1/2)*pi - ...`, `acosh(0*I)` in I was `(1/2)*(pi*I)`, `acosh(-1) + I` was `pi*I + I`.
#[test]
fn b1s5_a_variable_named_pi_or_i_never_names_a_constant() {
    let mut failures = Vec::new();
    for row in VAR_ROWS {
        let r = on_stack(STACK, move || check_var_row(row));
        if let Err(m) = r {
            failures.push(format!("{} in {} at {}: {m}", row.0, row.1, row.2));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows failed:\n{}",
        failures.len(),
        VAR_ROWS.len(),
        failures.join("\n")
    );
}

/// The controls (both profiles): with the variable `pi` the input's `pi` is the variable, as at
/// HEAD (`sin(pi)` in pi at 0 is pi - pi^3/6), and the variable `x` keeps round 4's exact values
/// (`acos(0*x)` at 0 is pi/2, written with the constant pi).
#[test]
fn b1s5_controls_keep_their_values() {
    on_stack(STACK, || {
        let pi = Symbol::new("pi");
        let r =
            tw(&parse("sin(pi)"), &pi, &Expr::from(0), 3, REPL_TAYLOR_WORK).expect("sin(pi) in pi");
        // pi - pi^3/6 at pi = 1/3 is 1/3 - 1/162 = 53/162
        let at = r.substitute(&pi, &parse("1/3"));
        let v = eval::eval(&at).expect("a rational");
        assert!(
            eval::close(&v, &eval::parse_coefficient("53/162", "0"), 50),
            "{r}"
        );
        let r = tw(
            &parse("acos(0*x)"),
            &x(),
            &Expr::from(0),
            2,
            REPL_TAYLOR_WORK,
        )
        .expect("acos(0*x)");
        let v = eval::eval(&r).expect("a constant");
        assert!(
            eval::close(&v, &eval::parse_coefficient(ACOS_0, "0"), 50),
            "{r}"
        );
        assert!(mentions(&r, "pi"), "{r}: the exact value pi/2");
    });
}

// ---------------------------------------------------------------------------------------------
// Resources B1: consumers that rebuild the tree

/// K followed by `n` divisions by 3 (review 5's `cons/sinchain2535.txt` for n = 2 535).
fn div3_chain(n: usize) -> String {
    format!("{K}{}", "/3".repeat(n))
}

/// Run `g(&r)` on its own thread with an 8 MB stack and wait at most `secs` seconds: `Some` with
/// the value and the wall time when it returns in time, `None` on a timeout (the thread keeps
/// running until the test binary exits, and the caller fails the test).
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

type Consumer<'a> = (&'a str, fn(&Expr) -> usize);

/// The consumers of review 5 (resources B1): `simplify` of the output of `substitute(x, 2)`, of
/// `expand` and of `differentiate(x)` (the REPL's `diff` is the last one).
const REBUILDING_CONSUMERS: [Consumer<'static>; 3] = [
    ("simplify of substitute(x, 2)", |r| {
        format!(
            "{}",
            r.substitute(&Symbol::new("x"), &Expr::from(2)).simplify()
        )
        .len()
    }),
    ("simplify of expand", |r| {
        usize::from(r.expand().simplify() == *r)
    }),
    ("simplify of differentiate(x)", |r| {
        usize::from(r.differentiate(&Symbol::new("x")).simplify() == *r)
    }),
];

/// One input at a point and an order through `taylor_with_budget` (REPL budget, within 20 s):
/// refused, or each consumer within `secs` (and VmHWM under 512 MB when `memory`).
fn consumers_within(what: &str, f: &str, a: i64, order: usize, secs: f64, memory: bool) {
    let fs = f.to_string();
    let r = on_stack(8 << 20, move || {
        let t = Instant::now();
        let r = tw(&parse(&fs), &x(), &Expr::from(a), order, REPL_TAYLOR_WORK);
        (r, t.elapsed().as_secs_f64())
    });
    let (r, t) = r;
    assert!(t < BAR_SECONDS, "{what}: taylor {t:.1} s");
    let what = format!("{what} at {a}, order {order}");
    let r = match r {
        Err(e) => {
            eprintln!("{what}: refused ({e:?})");
            return;
        }
        Ok(r) => Arc::new(r),
    };
    for (name, g) in REBUILDING_CONSUMERS {
        match within(secs, r.clone(), g) {
            None => panic!("{what}: {name} did not finish within {secs} s"),
            Some((n, t)) => eprintln!("{what}: {name} {t:.3} s ({n})"),
        }
        if memory {
            if let Some(b) = vm_hwm_bytes() {
                assert!(b < BAR_BYTES, "{what}: {name}: VmHWM {} MB", b >> 20);
            }
        }
    }
}

/// Resources B1 (both profiles): `sin(K/3 x 2535)/(1-x)` (5 095 bytes) at 0, order 10, is refused,
/// or `simplify` of each consumer's output returns within 20 s (the same deadline in debug, so a
/// fix must bound the work, not only speed it up). Before the fix (release): 24.1 s, 23.7 s and
/// 27.1 s; in the debug REPL 207 s and 225 s.
#[test]
fn b1r5_consumers_of_a_kernel_source_result_are_bounded() {
    let f = format!("sin({})/(1-x)", div3_chain(2535));
    assert_eq!(f.len(), 5095);
    consumers_within("sin(K/3 x 2535)/(1-x)", &f, 0, 10, BAR_SECONDS, false);
}

/// Resources B1, the bar itself (release, 8 MB thread): the input at orders 10 and 190 (190 was the
/// largest accepted order) and the same chain over (2 - x) at 1, order 10 (`expand` distributes the
/// coefficient over the binomial terms of (x - 1)^k there): refused, or each consumer within 20 s
/// with VmHWM under 512 MB. Before the fix: 24.1 s, 23.7 s, 27.1 s at order 10 and 418.6 s for
/// `substitute` at order 190.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b1r5_consumers_of_kernel_source_results_release() {
    let f = format!("sin({})/(1-x)", div3_chain(2535));
    for order in [10, 190] {
        consumers_within("sin(K/3 x 2535)/(1-x)", &f, 0, order, BAR_SECONDS, true);
    }
    let g = format!("sin({})/(2-x)", div3_chain(2535));
    consumers_within("sin(K/3 x 2535)/(2-x)", &g, 1, 10, BAR_SECONDS, true);
}

/// Resources B1, the edge (release, 8 MB thread): for the `/3` chain at 0, order 10, the largest
/// chain length n that `taylor_with_budget` accepts, by bisection, and each consumer there within
/// 20 s with VmHWM under 512 MB. Written after the fix, as a guard.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b1r5_consumers_at_the_edge_of_the_fold_cap_release() {
    for (what, make, a) in [
        (
            "sin(K/3 x n)/(1-x)",
            (|n| format!("sin({})/(1-x)", div3_chain(n))) as fn(usize) -> String,
            0i64,
        ),
        (
            "sin(K/3 x n)/(2-x)",
            (|n| format!("sin({})/(2-x)", div3_chain(n))) as fn(usize) -> String,
            1,
        ),
    ] {
        let accepted = |n: usize| -> bool {
            let f = make(n);
            on_stack(8 << 20, move || {
                tw(&parse(&f), &x(), &Expr::from(a), 10, REPL_TAYLOR_WORK).is_ok()
            })
        };
        let (mut lo, mut hi) = (1usize, 2535usize);
        assert!(accepted(lo), "{what}: n = 1 refused");
        if accepted(hi) {
            lo = hi;
        } else {
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if accepted(mid) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
        }
        eprintln!("{what} at {a}, order 10: the largest accepted n is {lo}");
        consumers_within(what, &make(lo), a, 10, BAR_SECONDS, true);
    }
}

/// A guard written after the fixes (both profiles): the new caps count the literals a node builds,
/// not the literal leaves a result shares, so the largest results of the earlier rounds stay
/// values: `exp(x)` at order 8996 (9 000 rational leaves 1/k!; the REPL budget refuses it from
/// about 9150) and `sin((2^254+1)^4096)/(1-x)` at order 192 (the edge of the text cap, review 4).
/// An early version of the literal-word count of this round counted the leaves and refused
/// `exp(x)` from order 8000.
#[test]
fn b1r5_caps_keep_the_largest_earlier_results() {
    for (f, order) in [("exp(x)", 8996usize), ("sin((2^254+1)^4096)/(1-x)", 192)] {
        let r = on_stack(STACK.max(8 << 20), move || {
            tw(&parse(f), &x(), &Expr::from(0), order, REPL_TAYLOR_WORK).map(|_| ())
        });
        assert!(r.is_ok(), "{f} at order {order}: {r:?}");
    }
}
