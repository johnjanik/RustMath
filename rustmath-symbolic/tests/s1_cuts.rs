//! S1 request C-1 (docs/workstreams/m23_golay/REQUESTS.md, section 2): an argument that leaves a
//! branch cut.
//!
//! The variable is real (series.rs, module conventions). When the constant term c0 of the argument
//! of a branch function (log, a non-integer power, arcsin, arccos, arctan, arcsinh, arccosh,
//! arctanh) lies on an open cut and another coefficient takes the argument off the cut, the
//! expansion through the principal value at c0 (the formal series gp computes) is the function on
//! one side of the point or on neither: `log(-1 - I*x^2)` at 0 is off by 2 pi I for x > 0 and for
//! x < 0. The engine expands there only when it decides that every other coefficient of the
//! argument is a real multiple of the cut's direction (real for the cuts on the real axis, purely
//! imaginary for those of arctan and arcsinh); a decided coefficient off that direction is
//! `NotLaurent` with a message that the argument leaves a branch cut at the point, and an
//! undecided real constant is `UndecidedZero`.
//!
//! `data/s1_cuts.tsv` (made by `data/gen_s1_cuts.py`, gp 2.15.4 at realprecision 300 through
//! `data/s1_gp.py`) holds the five inputs of the C-1 table, their mirror images (I -> -I), the
//! real-coefficient controls on each open cut of each function (the four of the request first),
//! complex arguments off every cut, more arguments that leave a cut, and undecided constants. The
//! note column of each row has the two-sided oracle's verdict on gp's formal series
//! (`python3 -I s1_gp.py --two-sided s1_cuts.tsv`): the controls are the function at x0 and -x0;
//! each refused row except the mirrors that leave the cut to the side of the principal value
//! (four of the five mirrors and four `lv:` rows, whose formal series is the function on both
//! sides: the rule refuses them anyway, a finer rule may expand them later) is not.
//!
//! Every row goes through `series_expansion` and `taylor_with_budget` with the REPL budget and the
//! limit path, on a thread with the bar's stack (2 MB in debug, 8 MB in release).
//!
//! MAGMA Handbook ch. 49 (digest
//! `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`, §49.5 transcendental
//! functions); Knuth, TAOCP vol. 2, §4.7; Brent and Kung, J. ACM 25(4), 1978.

mod s1_common;

use s1_common::api::*;
use s1_common::check::{check_all, kind, on_stack, parse_rows, x, Kind, Row};

const DATA: &str = include_str!("data/s1_cuts.tsv");
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

/// The five inputs of the C-1 table and their mirror images, in the order of the request.
const C1_ROWS: [&str; 10] = [
    "log(-1 - I*x^2)",
    "sqrt(-1 - I*x^2)",
    "log(-1 + I*x)",
    "arctanh(2 + I*x^2)",
    "arcsin(3/2 + I*x^2)",
    "log(-1 + I*x^2)",
    "sqrt(-1 + I*x^2)",
    "log(-1 - I*x)",
    "arctanh(2 - I*x^2)",
    "arcsin(3/2 - I*x^2)",
];

#[test]
fn data_has_the_request_rows_first() {
    let all = parse_rows(DATA);
    for (i, t) in C1_ROWS.iter().enumerate() {
        assert_eq!(all[i].input, *t, "row {i}");
    }
    let controls = [
        "log(-1 - x^2)",
        "sqrt(-4 + x)",
        "arcsin(2 + x)",
        "arctanh(2 - x^2)",
    ];
    for (i, t) in controls.iter().enumerate() {
        assert_eq!(all[10 + i].input, *t, "control {i}");
    }
}

/// The C-1 table and its mirror images refuse with `NotLaurent`, and the message says that the
/// argument leaves a branch cut at the point (both entry points; the limit path is `Unknown`).
#[test]
fn c1_rows_refuse_with_the_branch_cut_message() {
    on_stack(STACK, || {
        let mut fails = Vec::new();
        for t in C1_ROWS {
            let f = parse(t);
            let a = parse("0");
            let r1 = se(
                &f,
                &x(),
                &ExpansionPoint::At(a.clone()),
                3,
                Some(REPL_TAYLOR_WORK),
            );
            let r2 = tw(&f, &x(), &a, 3, REPL_TAYLOR_WORK);
            for (what, r) in [
                ("series_expansion", r1.map(|s| s.to_expr())),
                ("taylor_with_budget", r2),
            ] {
                match r {
                    Err(e) if kind(&e) == Kind::NotLaurent => {
                        let m = e.to_string();
                        if !m.contains("leaves a branch cut at the point") {
                            fails.push(format!("{t}: {what}: message {m:?}"));
                        }
                    }
                    Err(e) => fails.push(format!("{t}: {what}: Err({e:?})")),
                    Ok(v) => fails.push(format!("{t}: {what}: Ok({v})")),
                }
            }
            if !matches!(lim(&f, &x(), &a), LimitResult::Unknown) {
                fails.push(format!("{t}: limit path not Unknown"));
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    });
}

#[test]
fn c1_table_and_mirrors() {
    check_all(rows("c1"), DIGITS, STACK);
}

/// Real-coefficient controls on each open cut of each branch function: a value, equal to gp's
/// series (which the two-sided oracle checks against the function at x0 and -x0).
#[test]
fn controls_on_a_cut_keep_their_value() {
    let r = rows("ctl:");
    assert!(r.len() >= 36, "the controls");
    check_all(r, DIGITS, STACK);
}

#[test]
fn complex_arguments_off_the_cuts_keep_their_value() {
    check_all(rows("off:"), DIGITS, STACK);
}

#[test]
fn arguments_that_leave_a_cut_are_not_laurent() {
    check_all(rows("lv:"), DIGITS, STACK);
}

#[test]
fn undecided_coefficients_on_a_cut_are_undecided() {
    check_all(rows("und:"), DIGITS, STACK);
}

/// S1 C-1 minor items: real arguments on a cut whose real class comes from the structure or the
/// enclosure of a constant: f^g with a variable exponent and a positive base (regressions review
/// m2: pi, sqrt(2), e by their structure, 1 + sin(1) by its enclosure), and a truncated argument
/// scaled by a constant that is real only by its enclosure (semantics review m3: log(-1)^2).
/// Before, all seven refused ("cannot decide"); HEAD-before-S1 returned their series.
#[test]
fn classes_decided_by_structure_or_enclosure() {
    let mut r = rows("pg:");
    r.extend(rows("cls:"));
    assert_eq!(r.len(), 7, "the rows");
    check_all(r, DIGITS, STACK);
}

/// The limit path: before C-1 the formal series made `(log(-1 - I*x^2) - log(-1 + I*x^2))/(I*pi)`
/// at 0 the series 2/pi x^2 + ..., and the limit path read `Finite(0)`; the true limit is -2
/// (gp: `(log(-1 - I*10^-9) - log(-1 + I*10^-9))/(I*Pi)` is -1.99999999936...). The limit with
/// error is `Unknown` too.
#[test]
fn limit_path_does_not_read_a_wrong_branch() {
    on_stack(STACK, || {
        let f = parse("(log(-1 - I*x^2) - log(-1 + I*x^2))/(I*pi)");
        let a = parse("0");
        assert!(matches!(lim(&f, &x(), &a), LimitResult::Unknown));
        // S1 C-1 minor items (semantics review m6): the limit with error is `Unknown` (its
        // expansion refuses), not only "not Finite(0)"
        let (l, _) = lwe(&f, &x(), &a, 3);
        assert!(matches!(l, LimitResult::Unknown), "{l:?}");
    });
}

/// Free parameters are generic (series.rs, module conventions): a constant term that depends on
/// one is off the cut for generic values (`log(y + I*x)` at 0 holds for y off (-oo, 0]), and a
/// coefficient that depends on one is not decided real (`log(-1 + y*x)` at 0 refuses).
#[test]
fn parameters_are_generic() {
    on_stack(STACK, || {
        let y = Symbol::new("y");
        let a = parse("0");
        let f = parse("log(y + I*x)");
        let s = se(
            &f,
            &x(),
            &ExpansionPoint::At(a.clone()),
            2,
            Some(REPL_TAYLOR_WORK),
        )
        .expect("log(y + I*x) at 0");
        let c0 = s.coefficient(0).expect("constant term");
        assert_eq!(c0.to_string(), "log(y)", "{c0}");
        assert!(c0.symbols().contains(&y));
        let g = parse("log(-1 + y*x)");
        match se(&g, &x(), &ExpansionPoint::At(a), 2, Some(REPL_TAYLOR_WORK)) {
            Err(e) => assert_eq!(kind(&e), Kind::NotLaurent, "{e}"),
            Ok(s) => panic!("log(-1 + y*x): Ok({})", s.to_expr()),
        }
    });
}

/// S1 C-1 minor items (regressions review m1): a constant term that is a single kernel of a
/// parameter (sin(a), log(a), exp(a), ...) is a nonconstant function of the parameter, so it avoids
/// a cut for generic complex values of it (the module conventions), as a nonconstant polynomial in
/// the parameters does. HEAD-before-S1 returned these generic series; the C-1 rule refused them
/// ("cannot decide"). Reference: gp's series at a = 3/7, the engine's coefficients with a = 3/7
/// substituted, to 50 digits (gp 2.15.4, realprecision 60:
/// `a = 3/7; X = a + h + O(h^3); vector(3, k, polcoef(log(sin(X)), k - 1, h))`, and so on).
/// Two rows of the review stay refused, each for a reason outside the cut rule's generic test:
/// `log(tan(x))` at a has the constant term sin(a) cos(a)^-1, a product of two kernels (not
/// accepted, as `sin(a)^2 + cos(a)^2` is not), and `arctan(sin(x))` at a passes the cut rule but
/// its integrand divides by 1 + sin(a)^2, whose zero test is undecided (the zero test's generic
/// rule covers polynomials in the parameters only, design section 2.2): `UndecidedZero`, as before
/// C-1.
#[test]
fn kernels_of_a_parameter_are_generic() {
    on_stack(STACK, || {
        use s1_common::eval::{close, eval, parse_coefficient, show};
        let rows: [(&str, [(&str, &str); 3]); 6] = [
            (
                "log(sin(x))",
                [
                    (
                        "-0.878099743433677870708754412730652305979798801317846558390874",
                        "0",
                    ),
                    (
                        "2.18869574718762827289022875292541670403123256856938642669572",
                        "0",
                    ),
                    (
                        "-2.89519453687860520740940125377885842330828097532058885731712",
                        "0",
                    ),
                ],
            ),
            (
                "log(cos(x))",
                [
                    (
                        "-0.0947939262445452596184405748990706456264807723903565257984988",
                        "0",
                    ),
                    (
                        "-0.456893106903941882359843016520048135187291778866955133105343",
                        "0",
                    ),
                    (
                        "-0.604375655568168432683430748832382239397662532895703062165463",
                        "0",
                    ),
                ],
            ),
            (
                "sqrt(cos(x))",
                [
                    (
                        "0.953708735240569490524558103442346241217655689723450190735189",
                        "0",
                    ),
                    (
                        "-0.217871473562746360545494228673139772811507112668761881272693",
                        "0",
                    ),
                    (
                        "-0.263313177425598177752301633596630624908775872660012157407820",
                        "0",
                    ),
                ],
            ),
            (
                "sqrt(log(x))",
                [
                    (
                        "0",
                        "0.920487838261431683164596565565041187246585334716241494915240",
                    ),
                    (
                        "0",
                        "-1.26744386853628008803586094625241763000803444640262957780875",
                    ),
                    (
                        "0",
                        "0.606096145952069335499396741114480573610835825157947093806367",
                    ),
                ],
            ),
            (
                "log(log(x))",
                [
                    (
                        "-0.165702980993696313080406805815397629934989680650416294796857",
                        "3.14159265358979323846264338327950288419716939937510582097494",
                    ),
                    (
                        "-2.75385250266893355858977241202347321854204177277362421985367",
                        "0",
                    ),
                    (
                        "-0.579023883447551871196543254004486090805298795160964585319181",
                        "0",
                    ),
                ],
            ),
            ("log(exp(x))", [("3/7", "0"), ("1", "0"), ("0", "0")]),
        ];
        let a = Symbol::new("a");
        let at = parse("3/7");
        let mut fails = Vec::new();
        for (t, want) in rows {
            let f = parse(t);
            let s = match se(
                &f,
                &x(),
                &ExpansionPoint::At(Expr::Symbol(a.clone())),
                2,
                Some(REPL_TAYLOR_WORK),
            ) {
                Ok(s) => s,
                Err(e) => {
                    fails.push(format!("{t} at a: Err({e})"));
                    continue;
                }
            };
            for (k, (re, im)) in want.iter().enumerate() {
                let w = if re.contains('/') {
                    eval(&parse(re)).expect("rational")
                } else {
                    parse_coefficient(re, im)
                };
                let c = match s.coefficient(k as i64) {
                    Some(c) => c.substitute(&a, &at),
                    None => Expr::from(0),
                };
                match eval(&c) {
                    Ok(g) if close(&g, &w, 50) => {}
                    Ok(g) => fails.push(format!("{t}: h^{k}: {} != {}", show(&g), show(&w))),
                    Err(e) => fails.push(format!("{t}: h^{k}: {c}: {e}")),
                }
            }
        }
        let refused = [
            ("log(tan(x))", Kind::NotLaurent),
            ("arctan(sin(x))", Kind::UndecidedZero),
        ];
        for (t, k) in refused {
            match se(
                &parse(t),
                &x(),
                &ExpansionPoint::At(Expr::Symbol(a.clone())),
                2,
                Some(REPL_TAYLOR_WORK),
            ) {
                Err(e) if kind(&e) == k => {}
                Err(e) => fails.push(format!("{t} at a: Err({e:?})")),
                Ok(s) => fails.push(format!("{t} at a: Ok({})", s.to_expr())),
            }
        }
        assert!(fails.is_empty(), "{}", fails.join("\n"));
    });
}

/// S1 C-1 minor items (regressions review m1): the guards of the generic rule. A constant term
/// made of two kernels of a parameter can be a constant in disguise (sin(a)^2 + cos(a)^2 - 2 is
/// -1, on the cut of log; its zero test is undecided, so log refuses with `UndecidedZero` before
/// the cut rule), and abs is not analytic (abs(y) - 2 lies on the cut for every |y| < 2, an open
/// set): neither is a value.
#[test]
fn the_generic_rule_keeps_its_guards() {
    on_stack(STACK, || {
        let cases = [
            ("log(sin(x)^2 + cos(x)^2 - 2 + I*(x - a))", "a"),
            ("log(abs(y) - 2 + I*x)", "0"),
        ];
        for (t, p) in cases {
            let f = parse(t);
            let r = se(
                &f,
                &x(),
                &ExpansionPoint::At(parse(p)),
                2,
                Some(REPL_TAYLOR_WORK),
            );
            match r {
                Err(e) if kind(&e) == Kind::NotLaurent => {
                    assert!(e.to_string().contains("cannot decide"), "{t}: {e}")
                }
                Err(e) if kind(&e) == Kind::UndecidedZero => {}
                Err(e) => panic!("{t} at {p}: Err({e:?})"),
                Ok(s) => panic!("{t} at {p}: Ok({})", s.to_expr()),
            }
        }
    });
}

/// S1 C-1 minor items (semantics review m2, regressions review m3): the text of `NotLaurent`
/// claims no decided negative, since the variant also covers the refusals of the cut rule (a
/// mirror, whose function is analytic, and an undecided case): it says that no Laurent
/// expansion was found. The body says which case it is.
#[test]
fn a_cut_refusal_does_not_read_as_a_decided_negative() {
    on_stack(STACK, || {
        let cases = [
            ("log(-1 + I*x^2)", "leaves a branch cut at the point"),
            ("log(-1 + x*exp(I*pi))", "cannot decide"),
            ("log(x)", "log"),
        ];
        for (t, body) in cases {
            let f = parse(t);
            match se(
                &f,
                &x(),
                &ExpansionPoint::At(parse("0")),
                2,
                Some(REPL_TAYLOR_WORK),
            ) {
                Err(e) if kind(&e) == Kind::NotLaurent => {
                    let m = e.to_string();
                    assert!(
                        m.starts_with("no Laurent expansion found at the point: "),
                        "{t}: {m}"
                    );
                    assert!(m.contains(body), "{t}: {m}");
                }
                Err(e) => panic!("{t}: Err({e:?})"),
                Ok(s) => panic!("{t}: Ok({})", s.to_expr()),
            }
        }
    });
}

/// S1 C-1 minor items (regressions review m4): the class of a kernel of a positive constant is
/// read from the classes of its argument's kernels (`KData::kpos`), not from an enclosure of the
/// whole argument, so a chain of n such kernels costs about n, not n^2. Before, `sin(x) +
/// sqrt^1400(2)` and 2 000 nested logs exhausted the REPL budget (about 1.2 s in release); the
/// code before C-1 returned them in milliseconds. Reference: the series of sin(x) at 0 plus the
/// constant, whose class is not read here (gp: `sin(x + O(x^3))` = x + O(x^3)).
#[test]
fn chains_of_positive_constants_keep_their_value() {
    on_stack(STACK, || {
        let n = 1400;
        let sq = format!("sin(x) + {}2{}", "sqrt(".repeat(n), ")".repeat(n));
        let lg = format!("x + {}3{}", "log(2 + ".repeat(2000), ")".repeat(2000));
        for t in [sq, lg] {
            let f = parse(&t);
            let r = se(
                &f,
                &x(),
                &ExpansionPoint::At(parse("0")),
                2,
                Some(REPL_TAYLOR_WORK),
            );
            let head: String = t.chars().take(30).collect();
            match r {
                Ok(s) => {
                    let c1 = s.coefficient(1).expect("h^1");
                    assert_eq!(c1.to_string(), "1", "{head}...");
                }
                Err(e) => panic!("{head}...: Err({e})"),
            }
        }
    });
}

/// Text of at most `max` bytes: `head`, then `item(k)` for k = 1, 2, ... joined by `sep`, then
/// `tail`.
fn long_text(
    head: &str,
    item: impl Fn(usize) -> String,
    sep: &str,
    tail: &str,
    max: usize,
) -> String {
    let mut s = String::from(head);
    let mut k = 1;
    loop {
        let it = item(k);
        let extra = if k == 1 { 0 } else { sep.len() };
        if s.len() + extra + it.len() + tail.len() > max {
            break;
        }
        if k > 1 {
            s.push_str(sep);
        }
        s.push_str(&it);
        k += 1;
    }
    s.push_str(tail);
    s
}

/// The narrowed bar (c) for the branch-cut rule: inputs of at most 100 KB whose constants the
/// rule encloses (`series_ball::cdecide`), through `series_expansion` and `taylor_with_budget`
/// with the REPL budget on an 8 MB thread in release, each in under 20 s. (1) One coefficient, a
/// sum of 6 739 kernels arcsin(k) (complex), on the cut of log: `NotLaurent`. (2) A coefficient
/// whose enclosure fails (10 098 named functions f(k)) at order 300: the coefficient check runs
/// over every stored term. (3) 3 744 applications of log on its cut, each with an exact argument
/// whose coefficient I*log(-1) = -pi is real only by its enclosure (one such term is the control
/// `ctl:log:7`): the sum exhausts the REPL budget (`BudgetExceeded`).
/// `cargo test --release -p rustmath-symbolic --test s1_cuts -- --ignored --test-threads=1`
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn cut_rule_within_the_bar_release() {
    let inputs: Vec<(String, usize)> = vec![
        (
            long_text(
                "log(-1 + I*x*(",
                |k| format!("arcsin({})", k + 1),
                " + ",
                "))",
                100_000,
            ),
            30,
        ),
        (
            long_text(
                "log(-1 + I*exp(x)*(",
                |k| format!("f({k})"),
                " + ",
                "))",
                100_000,
            ),
            300,
        ),
        (
            long_text(
                "",
                |k| format!("log(-{k} + x*I*log(-1))"),
                " + ",
                "",
                100_000,
            ),
            3,
        ),
    ];
    for (f, order) in inputs {
        assert!(f.len() <= 100_000);
        let head: String = f.chars().take(40).collect();
        let (secs, r1, r2) = on_stack(8 << 20, move || {
            let e = parse(&f);
            let a = parse("0");
            let t = std::time::Instant::now();
            let r1 = se(
                &e,
                &x(),
                &ExpansionPoint::At(a.clone()),
                order as i64,
                Some(REPL_TAYLOR_WORK),
            )
            .map(|_| ())
            .map_err(|e| kind(&e));
            let r2 = tw(&e, &x(), &a, order, REPL_TAYLOR_WORK)
                .map(|_| ())
                .map_err(|e| kind(&e));
            (t.elapsed().as_secs_f64(), r1, r2)
        });
        println!("{head}... order {order}: {secs:.3} s, {r1:?}, {r2:?}");
        assert!(secs < 20.0, "{head}...: {secs} s");
    }
}
