//! S1 implementation, fix-up round 4: review 4's blocker B1 (resources lens,
//! `fu2/s1impl/S1.review4-resources.md`) and the two minor items of the semantics lens
//! (`fu2/s1impl/S1.review4-semantics.md`), as regression tests written before the fixes. Blocker
//! B2 of the resources lens (the memory of `simplify`) is in `s1_review4_memory.rs`, a test binary
//! of its own, because it measures the peak memory of the process.
//!
//! - B1: a kernel made from an input node renders as that node, and `simplify` folds the node's
//!   literal arithmetic left to right, while the engine evaluates the same `* /` chain by pairwise
//!   rounds (`Konst::product_many`). Neither result cap saw the difference: the text cap counts
//!   the folded literal once and the size guard of `Pow` passes. Release, 8 MB thread, budget 1e8,
//!   order 1: `sin((2^254+1)^4096/3/.../3)/(1-x)` with 12 000 divisions (24 KB) was `Ok` in
//!   0.037 s, then `simplify` of the result took 35.2 s (52.2 s with `/7`; 34.4 s with `g(...)` in
//!   place of `sin`; 169.5 s with 30 000 divisions, 60 KB; 355 s in the REPL, debug).
//! - Semantics m1: under the owner's first rule an exact-constant argument at a branch point was
//!   evaluated to the kernels arcsin(+-1), arccos(-1), arccosh(-1) and arcsinh(+-I), which have no
//!   ball enclosure. So a sum that held one of them could not be decided nonzero once it was
//!   inverted, logged or rooted: `1/(3 + asin(1 + 0*x))`, `1/(x + 3 + asin(1))`,
//!   `log(x + 1 + asin(1))`, ... were `UndecidedZero` although every input is regular.
//! - Semantics m2 (a handoff, not fixed: the frozen design scopes it out and fourteen first-step
//!   fuzz rows assert the refusal): abs and sign of a series whose coefficients are real but not
//!   rational are `Unsupported` at regular points, where HEAD's `taylor` gave the right series:
//!   `abs(x - sqrt(2))` at 0, `abs(sin(x))` at 1, `abs(cos(x))` at 2, `abs(x + pi)*x` at 0,
//!   `sign(x - pi)` at 0, `sign(log(x))` at 3.
//!
//! The `#[ignore]` test is the release measurement:
//! `cargo test --release -p rustmath-symbolic --test s1_review4 -- --ignored --test-threads=1`.
//! The other tests run in both profiles: `cargo test -p rustmath-symbolic --test s1_review4`.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: the expansion of 1/(1-x) has every coefficient 1, here times a constant; §49.5:
//! the transcendental functions of a series, their inverses and the real functions abs and sign
//! of a series); S1 design §2.2 (the constant field and its zero test), §2.4 (the atan family,
//! abs and sign), §2.10 (`simplify`), §2.15 (structural caps) and S1_BAR (a) and (c). Reference
//! values: PARI/GP 2.15.4 (`/usr/bin/gp`) at realprecision 300, through
//! `tests/data/s1_gp.py` (`series(text, point, order)`: gp on `f(P + h + O(h^(order+31)))`).

mod s1_common;

use s1_common::api::*;
use s1_common::check::*;
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

/// K = (2^254+1)^4096: 1 040 385 bits, exactly 2^20 bits a priori, the engine's and `simplify`'s
/// largest exact power (`MAX_EXACT_POWER_BITS`, `MAX_FOLD_POWER_BITS`).
const K: &str = "(2^254+1)^4096";

/// K followed by `n` divisions by `d`, no spaces (review 4's `chainT/*.txt`).
fn div_chain(d: u32, n: usize) -> String {
    format!("{K}{}", format!("/{d}").repeat(n))
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

/// B1's inputs (bytes: 24 025, 24 025, 24 023, 60 023), each at 0, order 1.
fn b1_inputs() -> Vec<(String, String)> {
    vec![
        (
            "sin(K /3 x 12000)/(1-x)".into(),
            format!("sin({})/(1-x)", div_chain(3, 12_000)),
        ),
        (
            "sin(K /7 x 12000)/(1-x)".into(),
            format!("sin({})/(1-x)", div_chain(7, 12_000)),
        ),
        (
            "g(K /3 x 12000)/(1-x)".into(),
            format!("g({})/(1-x)", div_chain(3, 12_000)),
        ),
        (
            "g(K /3 x 30000)/(1-x)".into(),
            format!("g({})/(1-x)", div_chain(3, 30_000)),
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// B1: simplify of a result re-folds a kernel's input-node source

/// B1 (both profiles): each input through `taylor_with_budget` with the REPL budget at 0, order 1,
/// is refused (a structural limit or the budget), or `simplify` of its result returns within
/// 20 s (the bar; a debug build has the same deadline, so a fix must bound the work, not only
/// speed it up). Before the fix: `Ok`, then `simplify` 35 to 170 s in release and 355 s in debug.
/// The control `exp(x + C)` (C the 12 000-level chain) stays a value: its kernel exp(C) is built
/// from C's normal form, the folded literal, and its `simplify` has nothing to fold (review 4:
/// 0.000 s).
#[test]
fn b1r4_simplify_of_a_kernel_source_chain_is_bounded() {
    let mut failures: Vec<String> = Vec::new();
    for (what, f) in b1_inputs() {
        assert!(f.len() <= 100_000, "{} bytes", f.len());
        let r = on_stack(STACK, move || {
            tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK)
        });
        match r {
            Err(e) => match kind(&e) {
                Kind::LimitExceeded(_) | Kind::BudgetExceeded => {
                    eprintln!("{what}: refused ({e:?})")
                }
                k => failures.push(format!("{what}: unexpected refusal {k:?}")),
            },
            Ok(r) => match within(BAR_SECONDS, Arc::new(r), |r| r.simplify() == *r) {
                None => failures.push(format!(
                    "{what}: Ok, then simplify of the result did not finish within {BAR_SECONDS} s"
                )),
                Some((_, t)) => eprintln!("{what}: Ok, simplify {t:.3} s"),
            },
        }
    }
    // the control: a value, and its simplify returns at once
    let f = format!("exp(x + {})", div_chain(3, 12_000));
    let r = on_stack(STACK, move || {
        tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK)
    });
    match r {
        Ok(r) => {
            if within(BAR_SECONDS, Arc::new(r), |r| r.simplify() == *r).is_none() {
                failures.push("the control exp(x + C): simplify did not finish".into());
            }
        }
        Err(e) => failures.push(format!(
            "the control exp(x + C): expected a value, got {e:?}"
        )),
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A named consumer of a result.
type Consumer<'a> = (&'a str, fn(&Expr) -> usize);

/// B1, the bar itself (release, 8 MB thread): each input through `taylor_with_budget` with 1e8
/// within 20 s, then refused, or `simplify` of the result, `Display` of `simplify`'s output and
/// `Display` of the result each within 20 s with VmHWM under 512 MB. Before the fix: `simplify`
/// 35.2 s, 52.2 s, 34.4 s and 169.5 s.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b1r4_simplify_of_kernel_source_chains_release() {
    for (what, f) in b1_inputs() {
        let r = on_stack(8 << 20, move || {
            let t = Instant::now();
            let r = tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK);
            let secs = t.elapsed().as_secs_f64();
            assert!(secs < BAR_SECONDS, "taylor {secs:.1} s");
            r
        });
        let r = match r {
            Err(e) => {
                eprintln!("{what}: refused ({e:?})");
                continue;
            }
            Ok(r) => Arc::new(r),
        };
        let steps: [Consumer<'_>; 3] = [
            ("simplify", |r| usize::from(r.simplify() == *r)),
            ("Display of simplify", |r| format!("{}", r.simplify()).len()),
            ("Display", |r| format!("{r}").len()),
        ];
        for (name, g) in steps {
            match within(BAR_SECONDS, r.clone(), g) {
                None => panic!("{what}: {name} did not finish within {BAR_SECONDS} s"),
                Some((n, t)) => eprintln!("{what}: {name} {t:.3} s ({n})"),
            }
            if let Some(b) = vm_hwm_bytes() {
                assert!(b < BAR_BYTES, "{what}: {name}: VmHWM {} MB", b >> 20);
            }
        }
    }
}

/// A family of inputs: its name and its text at length n.
type Family = (&'static str, fn(usize) -> String);

/// B1, the edge of the fold cap (release, 8 MB thread): for each family, the largest chain length n
/// that `taylor_with_budget` (1e8, order 1) accepts, by bisection, and `simplify` of that result
/// and `Display` of its output each within 20 s with VmHWM under 512 MB. The edge is set by
/// `MAX_RESULT_FOLD_WORK` (2^31 units of the model, about 1 ns each in release:
/// `series_const::tests::fold_work_calibration`), so `simplify` there takes a few seconds at most.
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn b1r4_simplify_at_the_edge_of_the_fold_cap_release() {
    let families: [Family; 4] = [
        ("sin(K /3 x n)/(1-x)", |n| {
            format!("sin({})/(1-x)", div_chain(3, n))
        }),
        ("g(K /7 x n)/(1-x)", |n| {
            format!("g({})/(1-x)", div_chain(7, n))
        }),
        ("sin(pi*K /3 x n)/(1-x)", |n| {
            format!("sin(pi*{})/(1-x)", div_chain(3, n))
        }),
        ("sin(K /(2^61-1) x n)/(1-x)", |n| {
            format!("sin({K}{})/(1-x)", "/(2^61-1)".repeat(n))
        }),
    ];
    for (what, make) in families {
        let accepted = |n: usize| -> bool {
            let f = make(n);
            on_stack(8 << 20, move || {
                tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK).is_ok()
            })
        };
        // the largest accepted n in [1, 12 000]: the chains of 12 000 are refused (b1r4 above)
        let (mut lo, mut hi) = (1usize, 12_000usize);
        assert!(accepted(lo), "{what}: n = 1 refused");
        assert!(!accepted(hi), "{what}: n = {hi} accepted");
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if accepted(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let f = make(lo);
        let r = on_stack(8 << 20, move || {
            tw(&parse(&f), &x(), &Expr::from(0), 1, REPL_TAYLOR_WORK).expect("accepted")
        });
        let r = Arc::new(r);
        let steps: [Consumer<'_>; 2] = [
            ("simplify", |r| usize::from(r.simplify() == *r)),
            ("Display of simplify", |r| format!("{}", r.simplify()).len()),
        ];
        for (name, g) in steps {
            match within(BAR_SECONDS, r.clone(), g) {
                None => panic!("{what} at n = {lo}: {name} did not finish within {BAR_SECONDS} s"),
                Some((n, t)) => {
                    eprintln!("{what}: largest accepted n = {lo}: {name} {t:.3} s ({n})")
                }
            }
            if let Some(b) = vm_hwm_bytes() {
                assert!(b < BAR_BYTES, "{what}: {name}: VmHWM {} MB", b >> 20);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Semantics m1: the exact values f(c) at a branch point are decided

/// A value row: input, point, order, the coefficients (exponent, re, im; absent ones are 0). The
/// limit path must be `Unknown`: every constant term here is irrational (design §2.13 gives
/// `Finite` only for a rational constant term).
type V = (&'static str, &'static str, i64, &'static [Coef<'static>]);

fn run_values(rows: &'static [V]) {
    let n = rows.len();
    let mut failures = Vec::new();
    for &(f, a, order, c) in rows {
        let r = on_stack(STACK, move || {
            check_value(f, a, order, c, &LimWant::Unknown, 60)
        });
        if let Err(m) = r {
            failures.push(format!("{f} at {a}, order {order}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {n} rows failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// gp: `1/(3 + asin(1))` = 1/(3 + Pi/2).
const INV_3_PI_2: &str = "0.2187802580784021380389378750354757360557384407832583944232129951153448";

/// Semantics m1 (both profiles): the value f(c) of an exact-constant argument at a branch point
/// can be inverted, logged and rooted. The owner's first rule asks for a value at every
/// exact-constant argument, and every input here is regular at 0. Before the fix each was
/// `UndecidedZero` (the kernels arcsin(1), arccos(-1), arccosh(-1), arcsinh(+-I) have no ball
/// enclosure, so the sums 3 + arcsin(1), 1 + arccosh(-1), ... were undecided). The values are
/// exact: arcsin(+-1) = +-pi/2, arccos(-1) = pi, arccos(0) = pi/2, arccosh(-1) = i pi,
/// arccosh(0) = i pi/2, arcsinh(+-i) = +-i pi/2. gp references (`tests/data/s1_gp.py`
/// `series(text, '0', 2)` on the simplified constant):
///   `1/(3 + asin(1))`, `1/(x + 3 + asin(1))`, `log(x + 1 + asin(1))`, `sqrt(x + asin(1) + 1)`,
///   `1/(exp(x) + asin(1))`, `1/(x + 2 + acos(-1))`, `1/(x + 1 + acosh(-1))`,
///   `1/(x + 1 + asinh(I))`, `1/(x + 3 + asin(-1))`, `1/(x + 1 + asinh(-I))`,
///   `1/(x + 3 + acos(0))`, `1/(x + 1 + acosh(0))`.
const M1_ROWS: &[V] = &[
    ("1/(3 + asin(1 + 0*x))", "0", 2, &[(0, INV_3_PI_2, "0")]),
    (
        "1/(3 + asin(-1*x^0))",
        "0",
        2,
        &[(
            0,
            "0.6996903371773598474048611342069184946654707038614477106250334170366327",
            "0",
        )],
    ),
    ("1/(3 + arcsin(x^0))", "0", 2, &[(0, INV_3_PI_2, "0")]),
    (
        "1/(x + 3 + asin(1))",
        "0",
        2,
        &[
            (0, INV_3_PI_2, "0"),
            (
                1,
                "-0.04786480132485224398196777994154295067946112213022347363094150670755337",
                "0",
            ),
            (
                2,
                "0.01047187358672261851093126035816686919006444346286899500906662878478044",
                "0",
            ),
        ],
    ),
    (
        "log(x + 1 + asin(1))",
        "0",
        2,
        &[
            (
                0,
                "0.9442157056960553917999443587348273917884374570814826181852002446337599",
                "0",
            ),
            (
                1,
                "0.3889845296483427106194046120460311382625773670439072047910832249575352",
                "0",
            ),
            (
                2,
                "-0.07565448215287120463104653648381166358865604508312563944578705880137093",
                "0",
            ),
        ],
    ),
    (
        "sqrt(x + asin(1) + 1)",
        "0",
        2,
        &[
            (
                0,
                "1.603370302455080592800022856384591819408816167296065266637225610623655",
                "0",
            ),
            (
                1,
                "0.3118431214763052580338696234713426002953834673279826531667177765904782",
                "0",
            ),
            (
                2,
                "-0.03032553748288290003062306524789751730714082038561380091650928953562649",
                "0",
            ),
        ],
    ),
    (
        "1/(exp(x) + asin(1))",
        "0",
        2,
        &[
            (
                0,
                "0.3889845296483427106194046120460311382625773670439072047910832249575352",
                "0",
            ),
            (
                1,
                "-0.1513089643057424092620930729676233271773120901662512788915741176027419",
                "0",
            ),
            (
                2,
                "-0.01679763584082411750990625242353839797986277334230900454654421879022403",
                "0",
            ),
        ],
    ),
    (
        "1/(x + 2 + acos(-1))",
        "0",
        2,
        &[
            (
                0,
                "0.1944922648241713553097023060230155691312886835219536023955416124787676",
                "0",
            ),
            (
                1,
                "-0.03782724107643560231552326824190583179432802254156281972289352940068546",
                "0",
            ),
            (
                2,
                "0.007357105789005885890142535507534158201099158967602079362405355001393363",
                "0",
            ),
        ],
    ),
    (
        "1/(x + 1 + acosh(-1))",
        "0",
        2,
        &[
            (
                0,
                "0.09199966835037523245580161148832061328711704948820710831848945743003368",
                "-0.2890254822222362424101704545685881578484520907366711335348031284032478",
            ),
            (
                1,
                "0.07507179039721716396484971272824465418694540181386500613396312715384293",
                "0.05318049701850601407713876239015029195462617232870839000626371379320259",
            ),
            (
                2,
                "-0.02227709861460476068540778189968829220933881775520524565574438660167861",
                "0.01680507233243166272642097770251045788309399680954734997769614279004817",
            ),
        ],
    ),
    (
        "1/(x + 1 + asinh(I))",
        "0",
        2,
        &[
            (
                0,
                "0.2884004391420009424309902578601556545360864835755433170280834008921228",
                "-0.4530183504502902067179420930737077646980763210440847380970741972880461",
            ),
            (
                1,
                "0.1220508125474029638486164522712084702225625171712010472918629542067426",
                "0.2613013824184971519946581560446232437618573248135425827802588156399917",
            ),
            (
                2,
                "-0.1535738291699170962712698349336361443906911973076085673513522161781992",
                "-0.02006817566656442983758356568473180798738253480713258974534116991563800",
            ),
        ],
    ),
    (
        "1/(x + 3 + asin(-1))",
        "0",
        2,
        &[
            (
                0,
                "0.6996903371773598474048611342069184946654707038614477106250334170366327",
                "0",
            ),
            (
                1,
                "-0.4895665679393675118334082403343576853303153971988759691843996777680308",
                "0",
            ),
            (
                2,
                "0.3425449969922589017058167026793884478817361411772820284554551284648239",
                "0",
            ),
        ],
    ),
    (
        "1/(x + 1 + asinh(-I))",
        "0",
        2,
        &[
            (
                0,
                "0.2884004391420009424309902578601556545360864835755433170280834008921228",
                "0.4530183504502902067179420930737077646980763210440847380970741972880461",
            ),
            (
                1,
                "0.1220508125474029638486164522712084702225625171712010472918629542067426",
                "-0.2613013824184971519946581560446232437618573248135425827802588156399917",
            ),
            (
                2,
                "-0.1535738291699170962712698349336361443906911973076085673513522161781992",
                "0.02006817566656442983758356568473180798738253480713258974534116991563800",
            ),
        ],
    ),
    (
        "1/(x + 3 + acos(0))",
        "0",
        2,
        &[
            (0, INV_3_PI_2, "0"),
            (
                1,
                "-0.04786480132485224398196777994154295067946112213022347363094150670755337",
                "0",
            ),
            (
                2,
                "0.01047187358672261851093126035816686919006444346286899500906662878478044",
                "0",
            ),
        ],
    ),
    (
        "1/(x + 1 + acosh(0))",
        "0",
        2,
        &[
            (
                0,
                "0.2884004391420009424309902578601556545360864835755433170280834008921228",
                "-0.4530183504502902067179420930737077646980763210440847380970741972880461",
            ),
            (
                1,
                "0.1220508125474029638486164522712084702225625171712010472918629542067426",
                "0.2613013824184971519946581560446232437618573248135425827802588156399917",
            ),
            (
                2,
                "-0.1535738291699170962712698349336361443906911973076085673513522161781992",
                "-0.02006817566656442983758356568473180798738253480713258974534116991563800",
            ),
        ],
    ),
];

#[test]
fn m1r4_exact_values_at_branch_points_are_decided() {
    run_values(M1_ROWS);
}

/// Semantics m1, the exact zero (both profiles): with the exact values, a divisor that is 0 only
/// through arcsin(1) - pi/2 or arccos(-1) - pi is the exact 0, so the input divides by an exact
/// zero (the owner's rule: `DivisionByZero` only then, and then always). Before the fix these
/// were the value `x/(arcsin(1) - pi/2)` (the documented convention for an undecided divisor of a
/// constant factor) and `UndecidedZero`. The limit path is `Unknown`.
#[test]
fn m1r4_an_exact_zero_through_the_values_is_a_division_by_zero() {
    let mut failures = Vec::new();
    for f in [
        "x/(asin(1) - pi/2)",
        "x/(acos(-1) - pi)",
        "1/(asinh(I) - I*pi/2)",
        "x/(acosh(-1) - I*pi)",
    ] {
        let r = on_stack(STACK, move || {
            check_refusal(f, "0", 2, Kind::DivisionByZero, &LimWant::Unknown)
        });
        if let Err(m) = r {
            failures.push(format!("{f}: {m}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ---------------------------------------------------------------------------------------------
// Semantics m2: abs and sign of a series with real coefficients (a handoff, not fixed here)

/// Semantics m2 (both profiles), recorded as a handoff in this round (`fu2/s1impl/S1_impl.md`,
/// fix-up round 4): abs and sign of a series whose coefficients are real but not all rational
/// stay the frozen design's typed refusal `Unsupported` (design §2.4, the abs/sign row: "all
/// coefficients rational"). Fourteen rows of the first step's fuzz data (`data/s1_fuzz.tsv`, the
/// `U:` rows with abs or sign) assert that refusal, and `R:y13.13` lists it among its outcomes, so
/// restoring HEAD's values is a change of the design's scope and of those rows, for the follow-up. This test pins the honest behaviour:
/// each input is refused with `Unsupported`, never a value of another kind or a wrong value.
/// The values the follow-up must give (gp, `tests/data/s1_gp.py` `series`, whose `AB` and `SG`
/// take the sign of the leading coefficient; HEAD's `taylor` gave them):
///   `abs(x - sqrt(2))` at 0, order 3: 1.41421356237309504880... - h;
///   `abs(sin(x))` at 1: 0.84147098480789650665... + 0.54030230586813971740... h - ...;
///   `abs(cos(x))` at 2: 0.41614683654714238699... + 0.90929742682568169539... h - ...;
///   `abs(x + pi)*x` at 0: 3.14159265358979323846... h + h^2;
///   `sign(x - pi)` at 0: -1; `sign(log(x))` at 3: 1.
/// The controls keep their outcome: a zero of odd order with rational coefficients is
/// `NotLaurent`, and abs of a series with a complex coefficient is `Unsupported`.
#[test]
fn m2r4_abs_and_sign_of_real_coefficients_stay_refused() {
    let mut failures = Vec::new();
    for (f, a, k) in [
        ("abs(x - sqrt(2))", "0", Kind::Unsupported),
        ("abs(sin(x))", "1", Kind::Unsupported),
        ("abs(cos(x))", "2", Kind::Unsupported),
        ("abs(x + pi)*x", "0", Kind::Unsupported),
        ("sign(x - pi)", "0", Kind::Unsupported),
        ("sign(log(x))", "3", Kind::Unsupported),
        ("abs(x)", "0", Kind::NotLaurent),
        ("sign(x - sqrt(2))", "sqrt(2)", Kind::NotLaurent),
        ("abs(x + I)", "0", Kind::Unsupported),
        ("abs(x + log(-2))", "0", Kind::Unsupported),
    ] {
        let r = on_stack(STACK, move || check_refusal(f, a, 3, k, &LimWant::Unknown));
        if let Err(m) = r {
            failures.push(format!("{f} at {a}: {m}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
