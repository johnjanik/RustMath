//! S1 regression tests, written before the fix (owner decision of 2026-11-15: "turn B1, B2 and the
//! red team's three remaining holes into regression tests before any fix is written").
//!
//! - Review 9, B1: arcsin, arccos, arccosh and arcsinh of an argument whose series is an exact
//!   constant at a branch point are a decided `DivisionByZero` in the prototype (the integrand is
//!   formed first and is singular there), although the value is the constant f(c). Every input row
//!   of review 9's table, its controls, and m2 (arctan(+-I), the pole that B1's fix needs) and m3
//!   (an inexact constant, x/x at a nonzero point).
//! - Review 9, B2: the pole rule for arctanh(+-1) fires on the constant term of a non-constant
//!   argument, so a logarithmic branch point is `DivisionByZero` instead of `NotLaurent`. Every input
//!   row of review 9's table and its controls; its last row, `atanh(exp(x-x))`, depends on an
//!   optional fold and accepts two outcomes, so a truncated argument without a fold,
//!   `atanh(1+(sin(x)^2+cos(x)^2-1))` (`PrecisionExhausted`), keeps that case of B2 caught.
//! - The three holes the red team still listed over the bar (`S1-symbolic.redteam3-series.md` H1,
//!   `S1-symbolic.redteam3-structure.md` H1 and H2; the folder README calls them the red team's last
//!   round): an empty truncated argument of negative precision taken as valuation >= 0; `0^r`
//!   decided from the syntax of the base and a rational test of r; results whose trees are
//!   exponential in their DAG, which make tree-walking consumers (simplify, substitute, PartialEq)
//!   exponential. Each with the red team's exact inputs and the expected outcome of the design's
//!   §4.1 tests `empty_argument_negative_precision`, `zero_power_rule` and `result_tree_cap`.
//! - The two holes of red team round 4 (`S1-symbolic.redteam4-series.md` H1 and H2, which the design
//!   closed in revision 7rt4): the precision sentinel that was also a legal exponent, and the units
//!   of a dense buffer that wrapped. Rows of the design's §4.1 tests `exact_sentinel` and
//!   `dense_buffer_units`.
//!
//! Scope of the owner's first rule: `s1_common/mod.rs`. A row whose outcome depends on a fold that
//! the scope leaves optional accepts each honest outcome (`check_outcomes`).
//!
//! The owner's two rules: an argument whose series is an exact constant c is evaluated (f(c), no
//! expansion, no integrand); a non-constant argument at a branch point is `NotLaurent`, never
//! `DivisionByZero`. `DivisionByZero` only when the input divides by an exact zero (or names a pole
//! of a constant, design §2.2 `apply`).
//!
//! Every outcome is checked through `series_expansion` and `taylor_with_budget` with the REPL
//! budget and through the limit path, on a thread with the bar's stack (2 MB in debug, 8 MB in
//! release). References: PARI/GP 2.15.4 (`gp -q`, realprecision 300), with the gp command in a
//! comment; `data/s1_gp.py` reproduces every gp value (`series(text, point, order)`).
//!
//! The budget tests of structure H2 and series H1 are `#[ignore]` and run in release:
//! `cargo test --release -p rustmath-symbolic --test s1_regressions -- --ignored --test-threads=1`
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4 precision model, §49.5 transcendental functions); Knuth, TAOCP vol. 2, §4.7;
//! Brent and Kung, J. ACM 25(4), 1978, 581-595.

mod s1_common;

use s1_common::api::*;
use s1_common::check::*;
use s1_common::eval;

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

/// gp: `asin(1)` (= Pi/2).
const PI_2: &str = "1.570796326794896619231321691639751442098584699687552910487472296153908";
/// gp: `asin(1) + 1`.
const PI_2_PLUS_1: &str = "2.570796326794896619231321691639751442098584699687552910487472296153908";
/// gp: `acosh(-1)` = Pi*I.
const PI: &str = "3.141592653589793238462643383279502884197169399375105820974944592307816";

#[derive(Clone, Copy)]
enum Lim {
    Unknown,
    Finite(&'static str),
}

impl Lim {
    fn want(self) -> LimWant {
        match self {
            Lim::Unknown => LimWant::Unknown,
            Lim::Finite(q) => LimWant::Finite(q.to_string()),
        }
    }
}

/// A value row: input, point, order, the nonzero coefficients (exponent, re, im), the limit path.
type V = (
    &'static str,
    &'static str,
    i64,
    &'static [Coef<'static>],
    Lim,
);
/// A refusal row: input, point, order, the kind, the limit path.
type R = (&'static str, &'static str, i64, Kind, Lim);

fn run_values(rows: &'static [V]) {
    let mut failures = Vec::new();
    for &(f, a, n, c, l) in rows {
        let r = on_stack(STACK, move || check_value(f, a, n, c, &l.want(), 60));
        if let Err(m) = r {
            failures.push(format!("{f} at {a}, order {n}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows failed:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

fn run_refusals(rows: &'static [R]) {
    let mut failures = Vec::new();
    for &(f, a, n, k, l) in rows {
        let r = on_stack(STACK, move || check_refusal(f, a, n, k, &l.want()));
        if let Err(m) = r {
            failures.push(format!("{f} at {a}, order {n}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} rows failed:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

// ------------------------------------------------------------------------------------------------
// Review 9, B1
// ------------------------------------------------------------------------------------------------

/// Review 9 B1, every row of its table (`EXP/review9/repro9.sh`, output `repro9.txt`: the
/// prototype proto13 and revision 8 give `DivisionByZero("0 to a negative power")` for each,
/// through the three entry points; HEAD's taylor gives `0/0` terms or a named function).
/// Expected: the value. gp references (on the simplified constant where gp's own series of the
/// exact-constant shape fails, as noted):
///   `asin(1)` = 1.5707963267948966192..., `acos(1)` = 0, `acosh(1)` = 0,
///   `acosh(-1)` = 3.1415926535897932385*I, `asinh(I)` = 1.5707963267948966192*I;
///   `X = h + O(h^4); X + 0*asin(X^0)` = h + O(h^4); `exp(X) + 0*acos(X^0)` = 1 + h + 1/2*h^2 + 1/6*h^3;
///   `asin((X-1)/(X-1))` at X = 1 + h is a gp error ("domain error in polcoef: degree > 0"), so the
///   reference is `asin(1) + X` = 2.5707963267948966192... + h;
///   `acosh(-X^0)` at X = 2 + h gives O(h^8) in gp (X^0 is a t_POL), so the reference is `acosh(-1)`.
/// The limit path: `Finite` only for a rational constant term (design §2.13).
const B1_ROWS: &[V] = &[
    ("arcsin(x^0)", "0", 2, &[(0, PI_2, "0")], Lim::Unknown),
    ("arccos(x^0)", "0", 2, &[], Lim::Finite("0")),
    (
        "x + 0*arcsin(x^0)",
        "0",
        3,
        &[(1, "1", "0")],
        Lim::Finite("0"),
    ),
    (
        "exp(x) + 0*arccos(x^0)",
        "0",
        3,
        &[
            (0, "1", "0"),
            (1, "1", "0"),
            (2, "1/2", "0"),
            (3, "1/6", "0"),
        ],
        Lim::Finite("1"),
    ),
    (
        "arccos(sign(x^2+1)) + x",
        "0",
        2,
        &[(1, "1", "0")],
        Lim::Finite("0"),
    ),
    (
        "x + arcsin(1 + 0*x)",
        "0",
        2,
        &[(0, PI_2, "0"), (1, "1", "0")],
        Lim::Unknown,
    ),
    (
        "arcsin((x-1)/(x-1)) + x",
        "1",
        2,
        &[(0, PI_2_PLUS_1, "0"), (1, "1", "0")],
        Lim::Unknown,
    ),
    ("arccosh(x^0)", "2", 1, &[], Lim::Finite("0")),
    ("arccosh(-x^0)", "2", 1, &[(0, "0", PI)], Lim::Unknown),
    ("arcsinh(I*x^0)", "0", 1, &[(0, "0", PI_2)], Lim::Unknown),
];

#[test]
fn b1_exact_constant_operand_is_evaluated() {
    run_values(B1_ROWS);
}

/// Review 9 B1, the controls of its required change 2: a pole of the constant stays
/// `DivisionByZero` (gp: `atanh(1)` "domain error in atanh: x = 1"; `atan(I)` "domain error in
/// atanh: x = -1"), and a non-constant argument at a branch point stays `NotLaurent`
/// (gp: `asin(1 - X^2)` at X = h is the formal series Pi/2 - sqrt(2)*h, one side of the principal
/// function Pi/2 - sqrt(2)*|x| + ..., which has no Laurent expansion; `acosh(X)` at X = 1 + h is
/// "domain error in sqrtn: valuation != Mod(0, 2)").
#[test]
fn b1_controls() {
    const ROWS: &[R] = &[
        ("arctanh(x^0)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("arctan(I*x^0)", "0", 1, Kind::DivisionByZero, Lim::Unknown),
        ("arcsin(1 - x^2)", "0", 2, Kind::NotLaurent, Lim::Unknown),
        ("arccosh(x)", "1", 2, Kind::NotLaurent, Lim::Unknown),
    ];
    run_refusals(ROWS);
}

/// Review 9 m2, which B1's fix needs (B1 required change 1: `apply` decides the poles of the
/// constant, arctanh(+-1) and arctan(+-I), so that `arctan(I*x^0)` stays `DivisionByZero` once an
/// exact-constant operand is evaluated). The prototype returns the kernel `arctan(I)` and the
/// limit path `Finite(0)` for `x*arctan(I)`. gp: `atan(I)` is "domain error in atanh: x = -1".
/// Expected: `DivisionByZero`, the limit path `Unknown`, never `Finite(0)`.
#[test]
fn b1_m2_arctan_at_plus_minus_i_is_a_pole() {
    const ROWS: &[R] = &[
        ("x*arctan(I)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("x*arctan(-I)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("x + arctan(I)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("arctan(I)", "0", 1, Kind::DivisionByZero, Lim::Unknown),
    ];
    run_refusals(ROWS);
}

/// Review 9 m3: `x/x` at a nonzero point is `1 + O(h^P)` (the inverse of an exact non-monomial is
/// truncated), so B1's rule does not see an exact constant, `1 - b^2` stays an empty truncated
/// series, and the prototype gives `PrecisionExhausted(131)`. That refusal is honest. The value
/// pi/2 (gp `asin(1)`) is correct too, should the port make x/x exact there. What the owner's rules
/// forbid is `DivisionByZero` or `NotLaurent` (the input has neither an exact zero nor a branch
/// point of a non-constant argument) or another value.
#[test]
fn b1_m3_inexact_constant_is_never_division_by_zero() {
    for (f, a) in [
        ("arcsin(x/x)", "1"),
        ("arcsin(x/x) + x", "2"),
        ("arccos(x/x)", "-1/2"),
    ] {
        let r = on_stack(STACK, move || {
            let fe = parse(f);
            let ae = parse(a);
            match se(
                &fe,
                &x(),
                &ExpansionPoint::At(ae),
                1,
                Some(REPL_TAYLOR_WORK),
            ) {
                Err(e) if kind(&e) == Kind::PrecisionExhausted => Ok(()),
                Err(e) => Err(format!(
                    "{f} at {a}: expected PrecisionExhausted or the value, got {e:?}"
                )),
                Ok(_) => {
                    let want: &[Coef<'_>] = match f {
                        "arcsin(x/x)" => &[(0, PI_2, "0")],
                        "arcsin(x/x) + x" => &[(0, "3.570796326794896619231321691639751442098584699687552910487472296153908", "0"), (1, "1", "0")],
                        _ => &[],
                    };
                    check_value(f, a, 1, want, &LimWant::Skip, 60)
                }
            }
        });
        r.unwrap_or_else(|m| panic!("{m}"));
    }
}

/// The owner's first rule away from the branch points: an atan-family function of an exact
/// constant is the exact constant f(c), so what follows sees an exact value. The prototype builds
/// the integrand there too and returns the truncated `O(h^P)` (`asin(0)` integrates the exact 0),
/// so the next operation cannot decide anything: `sqrt(arcsin(x-x))` is `PrecisionExhausted(132)`
/// and `log(arcsin(x-x))` asks for precision until the search ends. Expected: the value where the
/// constant is regular (gp on the simplified function, since gp's own series of these shapes is
/// "domain error in polcoef: t_SER = O(h^k)": sqrt(0) = 0, abs(0) = 0, 0^(1/3) = 0, and
/// `x + acos(1)^(1/2)` = x with acos(1) = 0), and `DivisionByZero` where the input then divides by
/// the exact zero (`log(asin(0))` = log(0), `1/atan(0)`, `1/acosh(1)`; gp: "domain error in log:
/// argument = 0", "impossible inverse"). The fuzz rows `B1:z3.7`, `B1:y14.59`, `B1:z1.43`,
/// `B1:z2.37`, `B1:z3.20` and `B1:y11.17` of `data/s1_fuzz.tsv` are the same class.
#[test]
fn b1_exact_constant_operand_at_a_regular_point() {
    const VALUES: &[V] = &[
        ("sqrt(arcsin(x-x))", "0", 2, &[], Lim::Finite("0")),
        ("abs(arctanh(x-x))", "0", 3, &[], Lim::Finite("0")),
        ("(arcsinh(x-x))^(1/3)", "0", 2, &[], Lim::Finite("0")),
        (
            "x + (arccos(x^0))^(1/2)",
            "0",
            2,
            &[(1, "1", "0")],
            Lim::Finite("0"),
        ),
    ];
    const REFUSALS: &[R] = &[
        (
            "log(arcsin(x-x))",
            "0",
            1,
            Kind::DivisionByZero,
            Lim::Unknown,
        ),
        ("1/arctan(x-x)", "0", 1, Kind::DivisionByZero, Lim::Unknown),
        ("1/arccosh(x^0)", "2", 1, Kind::DivisionByZero, Lim::Unknown),
    ];
    run_values(VALUES);
    run_refusals(REFUSALS);
}

// ------------------------------------------------------------------------------------------------
// Review 9, B2
// ------------------------------------------------------------------------------------------------

/// Review 9 B2, the first four rows of its table: proto13 gives `DivisionByZero("arctanh at 1 or
/// -1")` for each (revision 8: `NotLaurent("the function has a logarithmic branch point there")`).
/// Expected: `NotLaurent` for the four non-constant arguments with constant term +-1 (arctanh b =
/// (1/2) log((1 + b)/(1 - b)) has a logarithmic singularity there; gp: "domain error in intformal:
/// residue(series, pole) != 0" for `atanh(X)` at X = 1 + h, `atanh(1 + X)`, `X + 0*atanh(1 + X)` at
/// X = h and `cos(atanh(1 - X))` at X = 2 + h). The limit path `Unknown` for each.
///
/// `x + 0*atanh(1+x)` is x, and HEAD's taylor returns x; the refusal is intended (the owner's
/// second rule and design §2.9: each subexpression is expanded on its own, and `atanh(1 + x)` has
/// no Laurent expansion at 0), so it is not a regression in the sense of S1_BAR (b). The same holds
/// for the 23 `b:*:sub` rows and the 16 `a:*:1+0x` and `a:*:7+0x` rows of the singular-point
/// matrix (the list is in `s1_singular_points.rs`).
const B2_ROWS: &[R] = &[
    ("atanh(x)", "1", 2, Kind::NotLaurent, Lim::Unknown),
    ("atanh(1+x)", "0", 2, Kind::NotLaurent, Lim::Unknown),
    ("x + 0*atanh(1+x)", "0", 2, Kind::NotLaurent, Lim::Unknown),
    ("cos(atanh(1-x))", "2", 2, Kind::NotLaurent, Lim::Unknown),
    // found while writing B2_OPTIONAL_FOLD: a term far above the first working precision
    // (proto13: the same DivisionByZero; revision 8: NotLaurent; gp, X = h + O(h^40) or O(h^200):
    // "residue(series, pole) != 0")
    ("atanh(1+x^100)", "0", 2, Kind::NotLaurent, Lim::Unknown),
];

#[test]
fn b2_branch_point_of_a_non_constant_argument() {
    run_refusals(B2_ROWS);
}

/// Review 9 B2, the last row of its table: `atanh(exp(x - x))` at 0, order 2 (proto13:
/// `DivisionByZero("arctanh at 1 or -1")`, a decided claim without a decided exact zero; revision
/// 8: `PrecisionExhausted(131)`). Its outcome depends on a fold that the scope of the owner's rule
/// leaves optional (`s1_common/mod.rs`): exp of the exact 0 is `1 + O(h^P)` without the fold
/// (proto13: `1/(exp(x-x) - 1)` is `PrecisionExhausted(131)`; the engine does not know whether
/// b - 1 vanishes, so the honest answer is `PrecisionExhausted`; gp: "impossible inverse in
/// div_ser: O(h^3)"), and the exact 1 with it (then arctanh(1) is a pole of the constant,
/// `DivisionByZero`, as for the control `atanh(x^0)`). Both are accepted; the limit path is
/// `Unknown` for either. Every other result fails (a value, `NotLaurent`, any other kind).
const B2_OPTIONAL_FOLD: &[(&str, &str, i64)] = &[("atanh(exp(x-x))", "0", 2)];

/// The truncated case of B2 without an optional fold (found while writing `B2_OPTIONAL_FOLD`, which
/// accepts `DivisionByZero` and so no longer catches B2 on its own): `sin(x)^2 + cos(x)^2 - 1` is a
/// computed series whose every stored coefficient is 0, `O(h^P)` truncated (it is no exact constant
/// of the design's exact rules, the same argument as the series H1 control
/// `x^10*exp((sin(x)^2+cos(x)^2-1)/x^4)`), so the argument of arctanh is `1 + O(h^P)` and the engine
/// cannot decide whether b - 1 vanishes. proto13: `DivisionByZero("arctanh at 1 or -1")`, an
/// undecided answer reported as decided. Revision 8: `PrecisionExhausted(131)`. gp
/// (`X = h + O(h^40); atanh(1+(sin(X)^2+cos(X)^2-1))`): "impossible inverse in div_ser: O(h^41)".
/// Expected: `PrecisionExhausted`, the limit path `Unknown`.
const B2_TRUNCATED: &[R] = &[(
    "atanh(1+(sin(x)^2+cos(x)^2-1))",
    "0",
    2,
    Kind::PrecisionExhausted,
    Lim::Unknown,
)];

#[test]
fn b2_truncated_argument_at_a_pole_is_undecided() {
    run_refusals(B2_TRUNCATED);
}

#[test]
fn b2_argument_with_an_optional_fold() {
    for &(f, a, n) in B2_OPTIONAL_FOLD {
        let r = on_stack(STACK, move || {
            check_outcomes(
                f,
                a,
                n,
                &[
                    Outcome::Refusal(Kind::PrecisionExhausted, LimWant::Unknown),
                    Outcome::Refusal(Kind::DivisionByZero, LimWant::Unknown),
                ],
                60,
            )
        });
        r.unwrap_or_else(|m| panic!("{f} at {a}, order {n}: {m}"));
    }
}

/// `limit_with_error` on the rows of B1 and B2 (design §2.13: the limit, then for a finite limit
/// the `BigO` O(h^(order + 1)) when the expansion reaches order + 1, from an expansion that renders
/// no term; `Err(_)` gives `None`). The B2 rows include `B2_TRUNCATED` and `B2_OPTIONAL_FOLD`,
/// which is refused under either of its accepted outcomes. Its limit is the one `limit_via_series`
/// gives at order 5 (the engine under `LIMIT_SERIES_WORK`, S1 review 1 B2; the implementation step
/// called HEAD's `Expr::limit` there). The test asserts a `BigO` exactly when the limit is finite
/// and the expansion succeeds, of exponent order + 1; never a `BigO` for a row the engine refuses.
#[test]
fn limit_with_error_follows_the_expansion() {
    for &(f, a, n, _, _) in B1_ROWS {
        on_stack(STACK, move || {
            let (l, b) = lwe(&parse(f), &x(), &parse(a), n as usize);
            assert_eq!(
                b.is_some(),
                matches!(l, LimitResult::Finite(_)),
                "{f} at {a}: {l:?}, {:?}",
                b.as_ref().map(big_o_exponent)
            );
            if let Some(b) = b {
                assert_eq!(big_o_exponent(&b), Some(n + 1), "{f} at {a}");
            }
        });
    }
    let refused = B2_ROWS
        .iter()
        .chain(B2_TRUNCATED)
        .map(|&(f, a, n, _, _)| (f, a, n))
        .chain(B2_OPTIONAL_FOLD.iter().copied());
    for (f, a, n) in refused {
        on_stack(STACK, move || {
            let (l, b) = lwe(&parse(f), &x(), &parse(a), n as usize);
            assert!(
                b.is_none(),
                "{f} at {a}: the engine refuses, but limit_with_error gave ({l:?}, {:?})",
                b.as_ref().map(big_o_exponent)
            );
        });
    }
}

/// Review 9 B2, the controls of its required change: a pole of an exact constant stays
/// `DivisionByZero` (gp: `atanh(1)` "domain error in atanh: x = 1").
#[test]
fn b2_controls() {
    const ROWS: &[R] = &[
        ("x + atanh(1)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("atanh(x^0)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
    ];
    run_refusals(ROWS);
}

// ------------------------------------------------------------------------------------------------
// Red team, series H1: an empty truncated argument of negative precision taken as valuation >= 0
// ------------------------------------------------------------------------------------------------

/// `S1-symbolic.redteam3-series.md` H1 (design §4.1 `empty_argument_negative_precision`): at the
/// first working precision `(sin(x) - x)/x^4` is `O(h^q)` with q < 0, and the prototype of revision
/// 7rt2 took it as valuation >= 0, so `x^10*exp(..)` was a decided `0` (and `1 + x + x^2` with the
/// `1/(1-x)` summand). The argument has a pole (it is -x^-1/6 + ...), so exp, sin, cos, sinh, cosh,
/// erf and `2^(..)` of it have essential singularities. Expected: `NotLaurent` for each, the limit
/// path `Unknown`. gp (`data/s1_gp.py`, X = P + h + O(h^33)): "domain error in exp: valuation < 0"
/// (sin, cos: "domain error in sin/cos: valuation < 0"; erf, as `1 - erfc(z0) + 2/sqrt(Pi)*
/// intformal(exp(-z^2)*z')`, the exp error).
#[test]
fn rt_series_h1_empty_argument_negative_precision() {
    const ROWS: &[R] = &[
        (
            "x^10*exp((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*sin((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*cos((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*sinh((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*cosh((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*erf((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "x^10*2^((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "(1)/(1-x) + x^10*exp((sin(x)-x)/x^4)",
            "0",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
        (
            "(x-1)^10*exp((sin(x-1)-(x-1))/(x-1)^4)",
            "1",
            2,
            Kind::NotLaurent,
            Lim::Unknown,
        ),
    ];
    run_refusals(ROWS);
}

/// The same hole at infinity (`series_expansion` with `ExpansionPoint::Infinity`; h = 1/x):
/// `exp((sin(1/x) - 1/x)*x^4)/x^10`, the first row as a function of 1/x. Expected `NotLaurent`.
#[test]
fn rt_series_h1_at_infinity() {
    on_stack(STACK, || {
        let f = parse("exp((sin((1)/x)-(1)/x)*x^4)/x^10");
        match se(
            &f,
            &x(),
            &ExpansionPoint::Infinity,
            2,
            Some(REPL_TAYLOR_WORK),
        ) {
            Err(e) if kind(&e) == Kind::NotLaurent => {}
            Err(e) => panic!("expected NotLaurent, got {e:?}"),
            Ok(s) => panic!("expected NotLaurent, got Ok({})", s.to_expr()),
        }
    });
}

/// The limit path of series H1: the prototype of revision 7rt2 gave `Finite(0)`, `Finite(1)` (also
/// on a 2 MB debug thread) and `Finite(0)`. gp: f(t) = 1 + t^10*exp((sin(t^2) - t^2)/t^9) has
/// f(-0.1) = 2.41019254799853067716e62 and f(-0.05) = 1.11899537543893622045e566, so the left limit
/// is +infinity (`default(realprecision,30); f(t)=1+t^10*exp((sin(t^2)-t^2)/t^9); f(-0.1)`); the
/// last function has an essential singularity at 0. Expected: `Unknown`, never a finite value.
#[test]
fn rt_series_h1_limit_path_is_unknown() {
    for f in [
        "x^10*exp((sin(x^2)-x^2)/x^9)",
        "1 + x^10*exp((sin(x^2)-x^2)/x^9)",
        "x^30001*exp((exp(x^30000)-1)/x^30001)",
    ] {
        on_stack(STACK, move || {
            let r = lim(&parse(f), &x(), &parse("0"));
            assert!(
                matches!(r, LimitResult::Unknown),
                "{f}: expected Unknown, got {r:?}"
            );
        });
    }
}

/// Series H1 through a one-sided entry point (testcheck n9). Red team 3 series H1 noted that
/// `limit_enhanced` (`limits.rs:485`), whose series strategy calls `limit_via_series` for every
/// direction, also gave `Finite(1)` from the left. For the red team's own input,
/// `1 + x^10*exp((sin(x^2)-x^2)/x^9)`, HEAD's `limit_enhanced` decides before that strategy, at
/// direct substitution (`1 + 0^10*exp(0/0)` simplifies to 1; HEAD gives `Finite(1)` from the
/// left), which S1 does not change (design §2.13 changes `limit_via_series` and
/// `limit_with_error` only), so that input cannot test S1 here (tests_report.md, finding 7).
/// `x/x + x^10*exp((sin(x^2)-x^2)/x^9)` is the same function near 0, but its direct substitution
/// is `0/0`, so `limit_enhanced` reaches `limit_via_series`: HEAD gives `Unknown` from the left,
/// the prototype of revision 7rt2 (which had the hole) `Finite(1)`, proto13 `Unknown`. gp:
/// f(t) = 1 + t^10*exp((sin(t^2) - t^2)/t^9) has f(-0.1) = 2.41e62 and f(-0.05) = 1.12e566, so the
/// left limit is +infinity; the right limit is 1. Expected: from the left `Unknown` or `Infinity`
/// (the true limit), never `Finite`; from the right `Unknown` or `Finite(1)`.
#[test]
fn rt_series_h1_one_sided_limit() {
    on_stack(STACK, || {
        let f = parse("x/x + x^10*exp((sin(x^2)-x^2)/x^9)");
        let zero = parse("0");
        let left = lim_dir(&f, &x(), &zero, Direction::Left);
        assert!(
            matches!(left, LimitResult::Unknown | LimitResult::Infinity),
            "from the left: expected Unknown or Infinity, got {left:?}"
        );
        let right = lim_dir(&f, &x(), &zero, Direction::Right);
        let want = LimWant::AnyOf(vec![LimWant::Unknown, LimWant::Finite("1".into())]);
        assert!(
            limit_matches(&right, &want),
            "from the right: expected Unknown or Finite(1), got {right:?}"
        );
    });
}

/// Series H1 at the REPL's largest order (`TAYLOR_ORDER` = 20000): `x^30001*exp((exp(x^30000)-1)
/// /x^30001)` (38 bytes) was a decided `0` with precision 20001 in revision 7rt2. The argument is
/// x^-1 + O(x^29999) (exp(x^30000) - 1 = x^30000 + ...), so the function has an essential
/// singularity at 0. Expected: a refusal, never a value: `NotLaurent` (the design's §4.1 row), or
/// `PrecisionExhausted`/`BudgetExceeded` if the search for the pole's precision runs out first
/// (red team 3's fix sketch gave `PrecisionExhausted(134)` on the limit path).
#[test]
fn rt_series_h1_at_taylor_order() {
    on_stack(STACK, || {
        let f = parse("x^30001*exp((exp(x^30000)-1)/x^30001)");
        match tw(&f, &x(), &parse("0"), 20000, REPL_TAYLOR_WORK) {
            Err(e)
                if matches!(
                    kind(&e),
                    Kind::NotLaurent | Kind::PrecisionExhausted | Kind::BudgetExceeded
                ) => {}
            Err(e) => panic!(
                "expected a refusal NotLaurent, PrecisionExhausted or BudgetExceeded, got {e:?}"
            ),
            Ok(r) => panic!("expected a refusal, got the value {r}"),
        }
    });
}

/// The controls of series H1 (design §4.1): an argument that is empty at the first precision but
/// whose true valuation is >= 0 must still expand after the retry. gp: `exp((sin(X)-X)/X^2)` at
/// X = h is 1 - 1/6*h + 1/72*h^2; `X^10*exp((sin(X)^2+cos(X)^2-1)/X^4)` is O(h^3) at order 2 (the
/// argument is the exact 0 to every precision gp keeps), so the expansion to order 2 is 0.
/// `0^(sin(x)/x)` at 0 (orders 0 and 3) is the exact 0: r = 1 - x^2/6 + ... has a positive constant
/// term (red team 3 series H1, its last row; the prototype of 7rt2 said `NotLaurent` at order 0).
/// gp cannot raise the exact 0 to a series (`0^(sin(X)/X)`: "incorrect type in gpow(0,n)
/// (t_SER)"); the reference is 0^r = 0 for Re r(0) = 1 > 0.
#[test]
fn rt_series_h1_controls() {
    const ROWS: &[V] = &[
        (
            "exp((sin(x)-x)/x^2)",
            "0",
            2,
            &[(0, "1", "0"), (1, "-1/6", "0"), (2, "1/72", "0")],
            Lim::Finite("1"),
        ),
        (
            "x^10*exp((sin(x)^2+cos(x)^2-1)/x^4)",
            "0",
            2,
            &[],
            Lim::Finite("0"),
        ),
        ("0^((sin(x))/x)", "0", 0, &[], Lim::Finite("0")),
        ("0^((sin(x))/x)", "0", 3, &[], Lim::Finite("0")),
    ];
    run_values(ROWS);
}

// ------------------------------------------------------------------------------------------------
// Red team, structure H1: 0^r decided from the syntax of the base and a rational test of r
// ------------------------------------------------------------------------------------------------

/// `S1-symbolic.redteam3-structure.md` H1 (design §4.1 `zero_power_rule`): a base that mentions the
/// variable but whose series is the exact zero took the rule "both depend on the variable",
/// `exp(r log l)`, and `log` of the exact zero was `DivisionByZero("log(0)")`; and `0^r` with an
/// exponent whose constant term is positive but not rational was `NotLaurent`. The value is
/// `0^r = 0` wherever Re r(a) > 0. HEAD's taylor answers the first five rows correctly. gp cannot
/// raise an exact zero series to a series ("incorrect type in gpow(0,n) (t_SER)", "domain error in
/// gpow [irrational exponent]: valuation != 0"), so the references are gp's series of the
/// simplified functions: `exp(X)` and `cos(X)` at X = h (1 + h + 1/2*h^2 + 1/6*h^3; 1 - 1/2*h^2),
/// 0, 1, X at X = 1 + h (1 + h) and at X = h, and `-3^(1/3)` = -1.4422495703074083823... (the
/// random input of `gen3.py`: its base `(1+x)^(3^((2/3)^(1/3)))*0` is the exact zero, so the power
/// `0^x` at 2 is 0 and the input is 3^(1/3) * -(3^-0)).
#[test]
fn rt_structure_h1_zero_base_values() {
    const ROWS: &[V] = &[
        (
            "exp(x) + ((x+1)^2 - x^2 - 2*x - 1)^(x+1)",
            "0",
            3,
            &[
                (0, "1", "0"),
                (1, "1", "0"),
                (2, "1/2", "0"),
                (3, "1/6", "0"),
            ],
            Lim::Finite("1"),
        ),
        (
            "cos(x) + (x*0)^(1+x^2)",
            "0",
            3,
            &[(0, "1", "0"), (2, "-1/2", "0")],
            Lim::Finite("1"),
        ),
        ("(x-x)^x", "1", 2, &[], Lim::Finite("0")),
        (
            "1 + (x*0)^(2^x)",
            "1",
            2,
            &[(0, "1", "0")],
            Lim::Finite("1"),
        ),
        (
            "x + 0^(2^x)",
            "1",
            2,
            &[(0, "1", "0"), (1, "1", "0")],
            Lim::Finite("1"),
        ),
        (
            "x + 0^(x+sqrt(2))",
            "0",
            2,
            &[(1, "1", "0")],
            Lim::Finite("0"),
        ),
        (
            "3^(1/3)*+-3^-((1+x)^+3^--(2/3)^(1/3)*0)^x",
            "2",
            1,
            &[(
                0,
                "-1.442249570307408382321638310780109588391869253499350577546416194541688",
                "0",
            )],
            Lim::Unknown,
        ),
    ];
    run_values(ROWS);
}

/// Structure H1, the refusals of the design's rule for `0^r` (§2.5, revision 7rt3): decided from
/// the engine's own tests on r, never from a rational test. An undecided lead below exponent 0
/// (`(sin(1)^2+cos(1)^2-1)/x + 2`: the x^-1 coefficient is an undecided zero) is `UndecidedZero`; a
/// decided pole of r (`1/x`) or r(0) with real part decided 0 (`x`, `I + x`) is `NotLaurent`
/// (0^r has no expansion there); Re r(0) decided negative (`-1 + x`) is `DivisionByZero` (the input
/// divides by an exact zero, 0^-1); a constant term that cannot be decided (`y + x`, y a free
/// symbol) is `UndecidedZero`. gp: every row is "incorrect type in gpow(0,n) (t_SER)".
#[test]
fn rt_structure_h1_zero_power_refusals() {
    const ROWS: &[R] = &[
        (
            "x + 0^((sin(1)^2+cos(1)^2-1)/x + 2)",
            "0",
            2,
            Kind::UndecidedZero,
            Lim::Unknown,
        ),
        ("x + 0^(1/x)", "0", 2, Kind::NotLaurent, Lim::Unknown),
        ("x + 0^x", "0", 2, Kind::NotLaurent, Lim::Unknown),
        ("x + 0^(I + x)", "0", 2, Kind::NotLaurent, Lim::Unknown),
        ("x + 0^(-1 + x)", "0", 2, Kind::DivisionByZero, Lim::Unknown),
        ("x + 0^(y + x)", "0", 2, Kind::UndecidedZero, Lim::Unknown),
    ];
    run_refusals(ROWS);
}

// ------------------------------------------------------------------------------------------------
// Red team round 4, series H1 and H2 (design §4.1 `exact_sentinel` and `dense_buffer_units`)
// ------------------------------------------------------------------------------------------------

/// `S1-symbolic.redteam4-series.md` H1 (design §4.1 `exact_sentinel`, revision 7rt4): the sentinel
/// `EXACT = i64::MAX` was also a legal exponent and precision, so `(x^(2^62)*x^(2^62-1))/x^(2^63-1)`
/// was a decided 0 for the value 1 (the limit path `Finite(0)`), and
/// `(x^9223372036854775807 + 1)/x^9223372036854775807` lost its `+ 1`. Expected (§4.1): the values
/// 1, 1, `1/x^9223372036854775807 + 1` and the control 1, the limit path `Finite(1)` for the
/// value 1 and `Unknown` for the pole. References: gp cannot expand these
/// (`X = h + O(h^3); (X^(2^62)*X^(2^62-1))/X^(2^63-1)` is "overflow in valp()"), so the values are
/// the exponent sums in gp: `2^62 + (2^62-1) - (2^63-1)` = 0 and `9223372036854775806 + 1 -
/// 9223372036854775807` = 0 (the monomial x^0 = 1), and `(x^N + 1)/x^N` = x^-N + 1 with N = 2^63 - 1.
/// Each runs through `series_expansion`, `taylor_with_budget` and the limit path on the bar's stack.
#[test]
fn rt4_series_h1_exact_sentinel_values() {
    const ONE: &[Coef<'static>] = &[(0, "1", "0")];
    for f in [
        "(x^(2^62)*x^(2^62-1))/x^(2^63-1)",
        "(x^9223372036854775806*x)/x^9223372036854775807",
        "(x^4611686018427387903*x^4611686018427387903)/x^9223372036854775806",
    ] {
        on_stack(STACK, move || {
            check_value_sparse(
                f,
                &ExpansionPoint::At(parse("0")),
                0,
                ONE,
                &LimWant::Finite("1".into()),
                60,
            )
        })
        .unwrap_or_else(|m| panic!("{f} at 0, order 0: {m}"));
    }
    on_stack(STACK, || {
        check_value_sparse(
            "(x^9223372036854775807 + 1)/x^9223372036854775807",
            &ExpansionPoint::At(parse("0")),
            0,
            &[(-9223372036854775807, "1", "0"), (0, "1", "0")],
            &LimWant::Unknown,
            60,
        )
    })
    .unwrap_or_else(|m| panic!("(x^N + 1)/x^N at 0, order 0: {m}"));
    // at infinity (h = 1/x): h^(2^62) h^(2^62-1) h^-(2^63-1) = 1
    on_stack(STACK, || {
        check_value_sparse(
            "((1)/x^(2^62)*(1)/x^(2^62-1))*x^(2^63-1)",
            &ExpansionPoint::Infinity,
            0,
            ONE,
            &LimWant::Skip,
            60,
        )
    })
    .unwrap_or_else(|m| panic!("((1)/x^(2^62)*(1)/x^(2^62-1))*x^(2^63-1) at oo, order 0: {m}"));
}

/// `exact_sentinel`, the refusals (§4.1): inputs whose value needs an exponent or a precision
/// beyond i64. Expected: the honest refusal of the design, never a stored term or a value; the
/// limit path `Unknown`. `PrecisionExhausted` for `((1)/(1-x^N) - 1)/x^N` (N = 2^63 - 1), for the
/// two products that need the exponent 2^63, and for the power `(x+x^2)^(2^63-2)` (its value needs
/// a working precision above i64::MAX); `Unsupported("a series precision overflows i64")` for the
/// two clamped precisions. Revision 8 gave `Finite(0)` on the limit path for the limit 2^63 - 2 of
/// the last row (gp: `2^63 - 2` = 9223372036854775806 is that limit, the coefficient of x in
/// (1 + x)^(2^63 - 2)).
#[test]
fn rt4_series_h1_exact_sentinel_refusals() {
    const ROWS: &[R] = &[
        (
            "((1)/(1-x^9223372036854775807) - 1)/x^9223372036854775807",
            "0",
            0,
            Kind::PrecisionExhausted,
            Lim::Unknown,
        ),
        (
            "((1+x)*x^(2^62)*x^(2^62-1))/x^(2^63-1)",
            "0",
            2,
            Kind::PrecisionExhausted,
            Lim::Unknown,
        ),
        (
            "((x^(2^62)+x^(2^62+1))*(x^(2^62-1)+x^(2^62)))/x^(2^63-1)",
            "0",
            2,
            Kind::PrecisionExhausted,
            Lim::Unknown,
        ),
        (
            "(x^(2^63-4)*exp(x))/x^(2^63-2)",
            "0",
            2,
            Kind::Unsupported,
            Lim::Unknown,
        ),
        (
            "(x^(2^63-4)*((1)/(1-x)))/x^(2^63-2)",
            "0",
            2,
            Kind::Unsupported,
            Lim::Unknown,
        ),
        (
            "((x+x^2)^9223372036854775806)/x^9223372036854775806",
            "0",
            1,
            Kind::PrecisionExhausted,
            Lim::Unknown,
        ),
        (
            "(((x+x^2)^9223372036854775806)/x^9223372036854775806 - 1)/x",
            "0",
            0,
            Kind::PrecisionExhausted,
            Lim::Unknown,
        ),
    ];
    run_refusals(ROWS);
}

/// `S1-symbolic.redteam4-series.md` H2 (design §4.1 `dense_buffer_units`, revision 7rt4): the units
/// of a dense buffer of 2^60 slots or more wrapped, so the budget admitted the buffer and
/// `Vec::with_capacity` panicked ("capacity overflow"; debug: "attempt to add with overflow"), on
/// inputs HEAD answers in microseconds. Expected: `BudgetExceeded` with the REPL budget through
/// `series_expansion` and `taylor_with_budget` (the units saturate), the limit path `Unknown`, no
/// panic, on the bar's stack; every infinite recurrence divided by x^(2^k), k = 57 to 62; and
/// `exp(x)` through `series_expansion` at max_exponent 2^57 - 1, 2^60 - 1 and 2^63 - 2.
/// The recurrences are those §4.1 lists ("Miller and binomial powers, inv", with the design's own
/// sweep `EXP/rev7rt4/tools/rt4_extra.sh`): exp, sin, cosh, tan, log, atan, erf, a rational power,
/// `inv`, a product of two inverses, and the two paths of `pow_int` for n > 64 (§2.4 rule 7, which
/// never returns an exact power, so even the power of the polynomial 1 + x fills 2^k + 1 slots):
/// `(1+x)^100` and `(1+x)^1000` (Miller's recurrence, a decided constant term),
/// `(2+x+sin(1)^2+cos(1)^2-1)^1000` (a row of the design's sweep), and
/// `(sin(1)^2+cos(1)^2-1+x)^100` and `(x+sin(1)^2+cos(1)^2)^100` (the leads that §4.1
/// `large_power_symbolic_lead` gives for the binomial path). proto13 gives `BudgetExceeded` on each
/// in under 3 ms (release).
#[test]
fn rt4_series_h2_dense_buffer_units() {
    const ROWS: &[R] = &[
        (
            "exp(x)/x^(2^61)",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
        (
            "sin(x)/x^1152921504606846976",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
        (
            "(1+x)^(1/3)/x^1152921504606846976",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
        (
            "((1)/x^100000000000000000 + 1)^65",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
        (
            "(1)/(x^2305843009213693952+x^2305843009213693953)",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
        (
            "((1)/(1-x^9223372036854775806) - 1)/x^9223372036854775806",
            "0",
            0,
            Kind::BudgetExceeded,
            Lim::Unknown,
        ),
    ];
    run_refusals(ROWS);
    let mut failures = Vec::new();
    for g in [
        "exp(x)",
        "sin(x)",
        "cosh(x)",
        "tan(x)",
        "log(1+x)",
        "atan(x)",
        "erf(x)",
        "(1+x)^(1/3)",
        "(1+x+x^2)^(2/3)",
        "(1)/(1-x)",
        "(1)/(1-x)*(1)/(1+x)",
        "(1+x)^100",
        "(1+x)^1000",
        "(2+x+sin(1)^2+cos(1)^2-1)^1000",
        "(sin(1)^2+cos(1)^2-1+x)^100",
        "(x+sin(1)^2+cos(1)^2)^100",
    ] {
        for k in 57..=62 {
            let f = format!("({g})/x^(2^{k})");
            let r = on_stack(STACK, move || {
                check_refusal(&f, "0", 0, Kind::BudgetExceeded, &LimWant::Unknown)
                    .map_err(|m| format!("{f}: {m}"))
            });
            if let Err(m) = r {
                failures.push(m);
            }
        }
    }
    for n in [(1i64 << 57) - 1, (1i64 << 60) - 1, i64::MAX - 1] {
        let r = on_stack(STACK, move || {
            match se(
                &parse("exp(x)"),
                &x(),
                &ExpansionPoint::At(parse("0")),
                n,
                Some(REPL_TAYLOR_WORK),
            ) {
                Err(e) if kind(&e) == Kind::BudgetExceeded => Ok(()),
                Err(e) => Err(format!("exp(x) to {n}: expected BudgetExceeded, got {e:?}")),
                Ok(_) => Err(format!("exp(x) to {n}: expected BudgetExceeded, got Ok")),
            }
        });
        if let Err(m) = r {
            failures.push(m);
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `dense_buffer_units` without a budget (`series_expansion` with `None`): a buffer above
/// `MAX_SERIES_SLOTS` = 2^57 slots is `Unsupported("a series buffer of more than 2^57 slots")`
/// before any allocation (design §2.15), no panic.
#[test]
fn rt4_series_h2_dense_buffer_without_a_budget() {
    for f in ["exp(x)/x^(2^60)", "sin(x)/x^(2^61)"] {
        on_stack(STACK, move || {
            match se(&parse(f), &x(), &ExpansionPoint::At(parse("0")), 0, None) {
                Err(e) if kind(&e) == Kind::Unsupported => {}
                Err(e) => panic!("{f}: expected Unsupported, got {e:?}"),
                Ok(_) => panic!("{f}: expected Unsupported, got Ok"),
            }
        });
    }
}

// ------------------------------------------------------------------------------------------------
// Red team, structure H2: results whose trees are exponential in their DAG
// ------------------------------------------------------------------------------------------------

/// The red team's family: F_1 = (1+exp(x))^2, F_(k+1) = (1+exp(F_k))^2 (`inputs/nexp_k.txt` of
/// `EXP/redteam3-structure`; F_2 = `(1+exp((1+exp(x))^2))^2`).
fn f_k(k: usize) -> String {
    let mut s = "x".to_string();
    for _ in 0..k {
        s = format!("(1+exp({s}))^2");
    }
    s
}

/// The structural cap of the design (§2.15, revision 7rt3): `MAX_RESULT_TREE` = 2^21 nodes.
const MAX_RESULT_TREE: u64 = 1 << 21;

/// Structure H2 (design §4.1 `result_tree_cap`), the fast part: F_22, F_26 and F_30 at 1, order 0.
/// In revision 7rt2 the order-0 result of F_30 had a tree of 1.07e10 nodes (177 to 241 distinct),
/// and `==` with a clone took 28.7 s, `simplify` and `substitute` of F_22 aborted at 4 GiB (HEAD:
/// 111 to 151 nodes, microseconds). Expected: `Ok`, the coefficient is the input with x = 1 (no x,
/// a tree of at most 1000 nodes; the design measured 151 for F_30), and `simplify`, `substitute(x,
/// 2)` and `==` with a clone of the result return. The value itself overflows any float (F_3(1) is
/// about exp(2.04e12)), so it is checked as a structure. The limit path is `Unknown` (the constant
/// term is not rational, §2.13).
#[test]
fn rt_structure_h2_order_0_results_are_small() {
    for k in [22usize, 26, 30] {
        on_stack(STACK, move || {
            let f = parse(&f_k(k));
            let one = parse("1");
            let r =
                tw(&f, &x(), &one, 0, REPL_TAYLOR_WORK).unwrap_or_else(|e| panic!("F_{k}: {e:?}"));
            let t = tree_size(&r);
            assert!(t <= 1000, "F_{k} at 1, order 0: a tree of {t} nodes");
            assert!(
                !mentions(&r, "x"),
                "F_{k} at 1, order 0: the coefficient mentions x"
            );
            let _ = r.simplify();
            let _ = r.substitute(&x(), &parse("2"));
            let c = r.clone();
            assert!(c == r);
            assert!(matches!(lim(&f, &x(), &one), LimitResult::Unknown));
        });
    }
}

/// Structure H2, the values at small k (design §4.1 `result_tree_cap`: "coefficients equal to
/// gp's at k = 2 and 3"). gp: `X = 1 + h + O(h^2); (1+exp((1+exp(X))^2))^2` =
/// 1020426192336.7105045930204335957... + 41255128583453.453855847847676261...*h. F_3 at 1 is about
/// exp(2.04e12), beyond MPFR's default exponent range, so its coefficients are compared as log(c0)
/// and c1/c0 (`eval::eval_scaled`, which keeps the modulus of a value as its log). gp
/// (`default(realprecision, 300); X = 1 + h + O(h^2); F = (1+exp((1+exp((1+exp(X))^2))^2))^2;
/// c0 = polcoef(F, 0, h); c1 = polcoef(F, 1, h); printf("%.70g", log(c0)); printf("%.70g",
/// c1/c0)`): log(c0) = 2040852384673.421009186040867191494547958890783803198076072703889676490
/// (real; it is 2 F_2(1) to 70 digits) and c1/c0 =
/// 82510257166906.90771169569535252202833765340905692303065409870825854975. F_3's result is also
/// `Ok` with a tree under the cap.
#[test]
fn rt_structure_h2_small_k_values() {
    const ROWS: &[V] = &[(
        "(1+exp((1+exp(x))^2))^2",
        "1",
        1,
        &[
            (
                0,
                "1020426192336.710504593020433595747273979445391901599038036351944838245",
                "0",
            ),
            (
                1,
                "41255128583453.45385584784767626101416882670452846151532704935412927488",
                "0",
            ),
        ],
        Lim::Unknown,
    )];
    run_values(ROWS);
    on_stack(STACK, || {
        let f = parse(&f_k(3));
        let one = parse("1");
        let r = tw(&f, &x(), &one, 1, REPL_TAYLOR_WORK).expect("F_3 at 1, order 1");
        assert!(tree_size(&r) <= MAX_RESULT_TREE);
        let s = se(
            &f,
            &x(),
            &ExpansionPoint::At(one.clone()),
            1,
            Some(REPL_TAYLOR_WORK),
        )
        .expect("F_3 at 1, order 1");
        let c0 = s.coefficient(0).expect("c0");
        let c1 = s.coefficient(1).expect("c1");
        let v0 = eval::eval_scaled(&c0).unwrap_or_else(|e| panic!("c0 = {c0}: {e}"));
        let v1 = eval::eval_scaled(&c1).unwrap_or_else(|e| panic!("c1 = {c1}: {e}"));
        let log_c0 = v0.ln().expect("log(c0)");
        let want_log_c0 = eval::parse_coefficient(
            "2040852384673.421009186040867191494547958890783803198076072703889676490",
            "0",
        );
        assert!(
            eval::close(&log_c0, &want_log_c0, 60),
            "log(c0): engine {}, gp {}",
            eval::show(&log_c0),
            eval::show(&want_log_c0)
        );
        let ratio = (v1.ln().expect("log(c1)") - log_c0.clone()).exp();
        let want_ratio = eval::parse_coefficient(
            "82510257166906.90771169569535252202833765340905692303065409870825854975",
            "0",
        );
        assert!(
            eval::close(&ratio, &want_ratio, 60),
            "c1/c0: engine {}, gp {}",
            eval::show(&ratio),
            eval::show(&want_ratio)
        );
    });
}

/// Structure H2: F_12 at 1, order 1. Its expanded result names each kernel twice per level, a tree
/// of about 4.6e9 nodes (F_14: 2.68e9 in revision 7rt2). Expected: `LimitExceeded(TreeSize)`
/// before any rendering, through `series_expansion` and `taylor_with_budget`; the limit path
/// `Unknown`.
#[test]
fn rt_structure_h2_tree_cap_refuses() {
    on_stack(STACK, || {
        let f = f_k(12);
        check_refusal(
            &f,
            "1",
            1,
            Kind::LimitExceeded(StructureLimit::TreeSize),
            &LimWant::Unknown,
        )
        .unwrap_or_else(|m| panic!("F_12 at 1, order 1: {m}"));
    });
}

/// Structure H2: F_k at 1, order 1 through `taylor_with_budget` (REPL budget), then each consumer
/// of the result (`simplify`, `substitute(x, 2)`, `==` with a clone, `Display`, and the `Drop` of
/// the result and of every value made from it), all on one thread of `stack` bytes. Panics unless
/// the result is `Ok` with a tree of at most `MAX_RESULT_TREE` nodes; returns the seconds of the
/// expansion and of the consumers, and the tree and DAG sizes.
fn h2_result_and_consumers(k: usize, stack: usize) -> (f64, f64, u64, usize) {
    on_stack(stack, move || {
        let f = parse(&f_k(k));
        let t = std::time::Instant::now();
        let r = tw(&f, &x(), &parse("1"), 1, REPL_TAYLOR_WORK)
            .unwrap_or_else(|e| panic!("F_{k} at 1, order 1: expected Ok, got {e:?}"));
        let t_expand = t.elapsed().as_secs_f64();
        let n = tree_size(&r);
        assert!(n <= MAX_RESULT_TREE, "F_{k}: a tree of {n} nodes");
        let d = dag_size(&r);
        let t1 = std::time::Instant::now();
        let s = r.simplify();
        let u = r.substitute(&x(), &parse("2"));
        let c = r.clone();
        assert!(c == r);
        let text_len = format!("{r}").len();
        drop((s, u, c, r));
        let t_consumers = t1.elapsed().as_secs_f64();
        eprintln!(
            "F_{k}: taylor {t_expand:.3} s, tree {n}, dag {d}, consumers {t_consumers:.3} s, text {text_len} bytes"
        );
        (t_expand, t_consumers, n, d)
    })
}

/// Structure H2, the large results on the bar's debug stack (S1_BAR (c): a 2 MB thread in debug):
/// F_10 and F_11 at 1, order 1, whose results have about 4.6e5 and 1.0e6 nodes (design §4.1
/// `result_tree_cap`: both `Ok`), and every consumer of each result on the same 2 MB thread. The
/// bar sets no time in debug; the release test below times the same calls.
#[test]
fn rt_structure_h2_consumers_on_the_debug_stack() {
    for k in [10usize, 11] {
        h2_result_and_consumers(k, STACK);
    }
}

/// Run `g` on its own 8 MB thread (the bar's release stack), time it against the bar's 20 s, and
/// check VmHWM after it (the peak resident set of the whole process since it started: it covers
/// every call made so far in the process, the other ignored test included when both run, so it is
/// stricter than the bar's 512 MB per input). For the `#[ignore]` release tests, run with
/// `--test-threads=1` so that no other test runs at the same time.
fn timed_release(what: &str, g: impl FnOnce() -> Result<(), String> + Send + 'static) {
    let t0 = std::time::Instant::now();
    on_stack(8 << 20, g).unwrap_or_else(|m| panic!("{what}: {m}"));
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("{what}: {secs:.3} s");
    assert!(secs < BAR_SECONDS, "{what}: {secs:.1} s");
    if let Some(b) = vm_hwm_bytes() {
        assert!(b < BAR_BYTES, "{what}: VmHWM {} MB", b >> 20);
    }
}

/// Structure H2, the budget part (S1_BAR (c), release): F_10 and F_11 at 1, order 1, the largest
/// results under the cap (design §4.1 `result_tree_cap`: `Ok`, trees of 462 903 and 1 036 403
/// nodes in the prototype proto13; revision 7rt2: 1.04e7 nodes, `simplify` 0.83 s and 960 MB, and
/// F_12 aborted at 4 GiB), and F_12, which must be `LimitExceeded(TreeSize)` (any other result,
/// `Ok` included, fails). Each input goes through every public entry point with its own budget,
/// each call timed on its own 8 MB thread against the bar's 20 s, with VmHWM checked after it
/// (`timed_release`): `taylor_with_budget` with the REPL budget, followed by its consumers
/// (`simplify`, `substitute`, `==`, `Display`, `Drop`) on the same thread, so the time covers
/// them; `series_expansion` at max_exponent 1 with the REPL budget (`Ok` for F_10 and F_11,
/// `LimitExceeded(TreeSize)` for F_12, as in `rt_structure_h2_tree_cap_refuses`), including the
/// drop of its result; and the limit path (`limit_via_series`, its own budget `LIMIT_SERIES_WORK`),
/// `Unknown` for each (the constant term is not rational, §2.13).
/// `cargo test --release -p rustmath-symbolic --test s1_regressions rt_structure_h2_consumers_release -- --ignored --test-threads=1`
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn rt_structure_h2_consumers_release() {
    for k in [10usize, 11] {
        timed_release(
            &format!("F_{k} taylor_with_budget and consumers"),
            move || {
                let (te, tc, _, _) = h2_result_and_consumers(k, 8 << 20);
                eprintln!("F_{k}: taylor {te:.3} s, consumers {tc:.3} s");
                Ok(())
            },
        );
        timed_release(&format!("F_{k} series_expansion"), move || {
            match se(
                &parse(&f_k(k)),
                &x(),
                &ExpansionPoint::At(parse("1")),
                1,
                Some(REPL_TAYLOR_WORK),
            ) {
                Ok(_) => Ok(()),
                Err(e) => Err(format!("expected Ok, got {e:?}")),
            }
        });
        timed_release(&format!("F_{k} limit path"), move || {
            match lim(&parse(&f_k(k)), &x(), &parse("1")) {
                LimitResult::Unknown => Ok(()),
                other => Err(format!("expected Unknown, got {other:?}")),
            }
        });
    }
    timed_release("F_12 taylor_with_budget", || {
        match tw(&parse(&f_k(12)), &x(), &parse("1"), 1, REPL_TAYLOR_WORK) {
            Err(e) if kind(&e) == Kind::LimitExceeded(StructureLimit::TreeSize) => Ok(()),
            Err(e) => Err(format!("expected LimitExceeded(TreeSize), got {e:?}")),
            Ok(_) => Err("expected LimitExceeded(TreeSize), got Ok".into()),
        }
    });
    timed_release("F_12 series_expansion", || {
        match se(
            &parse(&f_k(12)),
            &x(),
            &ExpansionPoint::At(parse("1")),
            1,
            Some(REPL_TAYLOR_WORK),
        ) {
            Err(e) if kind(&e) == Kind::LimitExceeded(StructureLimit::TreeSize) => Ok(()),
            Err(e) => Err(format!("expected LimitExceeded(TreeSize), got {e:?}")),
            Ok(_) => Err("expected LimitExceeded(TreeSize), got Ok".into()),
        }
    });
    timed_release("F_12 limit path", || {
        match lim(&parse(&f_k(12)), &x(), &parse("1")) {
            LimitResult::Unknown => Ok(()),
            other => Err(format!("expected Unknown, got {other:?}")),
        }
    });
}

/// Series H1 at `TAYLOR_ORDER`, the budget part (release, 8 MB): the refusal must come within the
/// bar through each public entry point with its own budget, each timed on its own (red team 3
/// measured the wrong `0` in 4 ms with 6.6e5 units): `taylor_with_budget` at order 20000 and
/// `series_expansion` at max_exponent 20000, both with the REPL budget, and the limit path
/// (`limit_via_series`, its own budget `LIMIT_SERIES_WORK`). VmHWM is checked after each call
/// (`timed_release`: the peak of the whole process, stricter than the bar).
/// `cargo test --release -p rustmath-symbolic --test s1_regressions rt_series_h1_at_taylor_order_release -- --ignored --test-threads=1`
#[test]
#[ignore = "release budget test: run with --release -- --ignored --test-threads=1"]
fn rt_series_h1_at_taylor_order_release() {
    const F: &str = "x^30001*exp((exp(x^30000)-1)/x^30001)";
    timed_release("taylor_with_budget", || {
        match tw(&parse(F), &x(), &parse("0"), 20000, REPL_TAYLOR_WORK) {
            Err(_) => Ok(()),
            Ok(r) => Err(format!("expected a refusal, got the value {r}")),
        }
    });
    timed_release("series_expansion", || {
        match se(
            &parse(F),
            &x(),
            &ExpansionPoint::At(parse("0")),
            20000,
            Some(REPL_TAYLOR_WORK),
        ) {
            Err(_) => Ok(()),
            Ok(s) => Err(format!("expected a refusal, got the value {}", s.to_expr())),
        }
    });
    timed_release("limit path", || match lim(&parse(F), &x(), &parse("0")) {
        LimitResult::Unknown => Ok(()),
        other => Err(format!("expected Unknown, got {other:?}")),
    });
}
