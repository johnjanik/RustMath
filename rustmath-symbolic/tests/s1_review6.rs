//! S1 implementation, finish under the narrowed bar: two minor items of review 6
//! (`fu2/s1impl/S1.review6-semantics.md` m1 and m3), as regression tests written before the
//! fixes.
//!
//! - m1. A base of nonzero valuation to a truncated exponent whose only stored term is an integer
//!   constant term: `x^((1+x)/(1+x))` at 0 is `x^(1 + O(h^P))`. Whether the exponent is the
//!   integer 1 is undecided (the truncated part may be 0 or not), and x^1 is a Laurent series
//!   while x^(1 + r) with r not 0 is not. The module rule of `series.rs` ("whether an exponent is
//!   an integer ... an undecided test is reported") asks for a refusal that decides nothing. The
//!   engine took log of the base and reported `NotLaurent` ("log has a logarithmic singularity at
//!   the point"), a decided answer for an input that is x (review 6, class H: 18 rows, the true
//!   functions are 1, x and x^2). Expected now: `PrecisionExhausted` (the driver raises the
//!   precision and never decides), or `UndecidedZero` when the constant term is a constant whose
//!   integrality the field cannot decide. A decided non-integer constant term, or another stored
//!   term with a decided nonzero coefficient, keeps `NotLaurent` (fix-up round 1 of the finish:
//!   another stored term with an undecided coefficient is `UndecidedZero`, `tests/s1_finish1.rs`).
//! - m3. A Gaussian constant A + B*I (A, B free of I) is inverted exactly as (A - B*I)/(A^2 + B^2)
//!   when A^2 + B^2 is decided nonzero. Before, the inverse of a multi-term constant was the
//!   kernel 1/(A + B*I), which has no ball enclosure when it is complex, so
//!   `1/(x + 1/(1+I) - (1-I)/2)` at 0 was `UndecidedZero` although the constant is 0 (review 6
//!   m3; 101 rows of its random class were undecided on such constants).
//!
//! S1_BAR (a): an undecided answer reported as decided (m1). m3 is a refusal that the exact
//! inverse decides. Run: `cargo test -p rustmath-symbolic --test s1_review6` (both profiles).
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.1.3-49.1.4: the precision model, an O-term is not an exact zero; §49.5: powers and the
//! transcendental functions of a series); S1 design §2.1 (truncated and exact series), §2.2 (the
//! constant field and its zero test), §2.5 (`Pow`). Reference values: PARI/GP 2.15.4
//! (`/usr/bin/gp`), `\p 70`, the script `fu2/s1impl/fin/refs6.gp` (run as `gp -q < refs6.gp`):
//! `(1+x)^((1+x)/(1+x)) + O(x^4)` is `1 + x`, `exp(x)^(1/(1+x)*(1+x)) + O(x^4)` is
//! `1 + x + 1/2*x^2 + 1/6*x^3`, `1/(1+I) - (1-I)/2` is 0, `(9/8*I)/(1+I)^2` is 9/16,
//! `1/(x + 1/(1+I) - (1-I)/2) + O(x^3)` is `x^-1`, `1/(x + 1/(2+3*I)) + O(x^3)` is
//! `(2 + 3*I) + (5 - 12*I)*x + (-46 + 9*I)*x^2`, `1/(x + 1/(Pi+I)) + O(x^3)` and
//! `log(x + 1/(1+I)) + O(x^3)` as below.

mod s1_common;

use s1_common::check::*;

const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};
const DIGITS: u32 = 60;

/// A named check, run on its own thread.
type Case = (String, Box<dyn FnOnce() -> Result<(), String> + Send>);

/// Run each check on a thread with the bar's stack and collect the failures.
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

/// A refusal row: input, point, order, the accepted kinds (each with the limit path `Unknown`).
fn refusal(f: &str, a: &str, order: i64, kinds: &[Kind]) -> Case {
    let (f, a, kinds) = (f.to_string(), a.to_string(), kinds.to_vec());
    (
        format!("{f} at {a}"),
        Box::new(move || {
            let outs: Vec<Outcome<'_>> = kinds
                .iter()
                .map(|k| Outcome::Refusal(*k, LimWant::Unknown))
                .collect();
            check_outcomes(&f, &a, order, &outs, DIGITS)
        }),
    )
}

/// A value row: input, point, order, gp's coefficients, the limit path.
fn value(f: &str, a: &str, order: i64, want: &'static [Coef<'static>], lim: LimWant) -> Case {
    let (f, a) = (f.to_string(), a.to_string());
    (
        format!("{f} at {a}"),
        Box::new(move || check_value(&f, &a, order, want, &lim, DIGITS)),
    )
}

/// The six truncated zeros of review 6 (class H): each is identically 0, and its series is
/// `O(h^P)` (an empty truncated series).
const TRUNCATED_ZEROS: [&str; 6] = [
    "sin(x)^2+cos(x)^2-1",
    "(1+x)/(1+x)-1",
    "exp(x)*exp(-x)-1",
    "log(exp(x))-x",
    "sqrt(1+x)^2-1-x",
    "tan(x)*cos(x)-sin(x)",
];

/// m1: the 18 class-H rows of review 6 (`x^(x^-5*Z)`, `x^(Z)` and `x^(Z/x^2)` at 0 for the six
/// truncated zeros Z; each function is x^0 = 1) and the hand rows of its section 12 (functions x,
/// x^2 and 1/x). Each exponent is a truncated series whose only stored term is an integer constant
/// term, or no term at all, and each base has a decided nonzero valuation. Before: `NotLaurent`
/// on every row. Now `PrecisionExhausted`.
#[test]
fn m1r6_a_truncated_integer_exponent_is_undecided() {
    let pe = [Kind::PrecisionExhausted];
    let mut cases = Vec::new();
    for z in TRUNCATED_ZEROS {
        for shape in ["x^(x^-5*(%))", "x^((%))", "x^((%)/x^2)"] {
            cases.push(refusal(&shape.replace('%', z), "0", 2, &pe));
        }
    }
    for f in [
        // review 6, section 12, m1
        "x^((1+x)/(1+x))",
        "x^(sin(x)^2+cos(x)^2)",
        "x^(2*(sin(x)^2+cos(x)^2))",
        "x^(2*cos(x-x))",
        "x^(exp(x)*exp(-x))",
        // review 6, hand rows `xr.tsv`
        "x^((1+x)/(1+x)-1)",
        "(x^2)^((2+x)/(2+x))",
        "x^(cos(x-x))",
        "(sin(x))^(cos(x)^2+sin(x)^2)",
        "x^(sin(x)/sin(x))",
        // a base of negative valuation: the same rule (the function is 1/x)
        "(1/x)^((1+x)/(1+x))",
        "(1/sin(x))^(exp(x)*exp(-x))",
    ] {
        cases.push(refusal(f, "0", 3, &pe));
    }
    run(cases);
}

/// m1: the outcomes the rule must not change. An exact exponent takes the constant-exponent rule
/// (a value); a base of valuation 0 is regular (a value); a decided non-integer constant term, or
/// a stored term other than the constant term with a decided nonzero coefficient, is
/// `NotLaurent`; a constant term whose integrality
/// the constant field cannot decide (sin(1)^2 + cos(1)^2, which is 1) is `UndecidedZero`, never
/// `NotLaurent`.
#[test]
fn m1r6_controls_keep_their_outcomes() {
    let x1: &'static [Coef<'static>] = &[(1, "1", "0")];
    let one_plus_x: &'static [Coef<'static>] = &[(0, "1", "0"), (1, "1", "0")];
    let exp_x: &'static [Coef<'static>] = &[
        (0, "1", "0"),
        (1, "1", "0"),
        (2, "1/2", "0"),
        (3, "1/6", "0"),
    ];
    let fin = |q: &str| LimWant::Finite(q.to_string());
    let cases = vec![
        // exact exponents: x (gp: `x + O(x^4)` for the simplified input)
        value("x^(x-x+1)", "0", 3, x1, fin("0")),
        value("x^(x/x)", "0", 3, x1, fin("0")),
        value("x^(1+0*sin(x))", "0", 3, x1, fin("0")),
        value("x^((x+1)^2-x^2-2*x)", "0", 3, x1, fin("0")),
        // a base of valuation 0 (gp `(1+x)^((1+x)/(1+x)) + O(x^4)`, `exp(x)^(1/(1+x)*(1+x))`)
        value("(1+x)^((1+x)/(1+x))", "0", 3, one_plus_x, fin("1")),
        value("x^((1+x)/(1+x))", "1", 3, one_plus_x, fin("1")),
        value("exp(x)^(1/(1+x)*(1+x))", "0", 3, exp_x, fin("1")),
        // NotLaurent: a non-integer constant term, or a term in h
        refusal("x^(1+x)", "0", 3, &[Kind::NotLaurent]),
        refusal("x^(x + (1+x)/(1+x))", "0", 3, &[Kind::NotLaurent]),
        refusal("x^(1/2 + (1+x)/(1+x) - 1)", "0", 3, &[Kind::NotLaurent]),
        refusal("x^(sqrt(2) + (1+x)/(1+x) - 1)", "0", 3, &[Kind::NotLaurent]),
        // undecided integrality: an honest refusal that decides nothing
        refusal(
            "x^(sin(1)^2+cos(1)^2 + (1+x)/(1+x) - 1)",
            "0",
            3,
            &[Kind::UndecidedZero],
        ),
    ];
    run(cases);
}

/// m3: a Gaussian constant is inverted exactly, so a constant that is 0 through such an inverse
/// is the exact 0. `1/(1+I) = (1-I)/2`, `(9/8*I)/(1+I)^2 = 9/16`, `1/(pi+I) = (pi-I)/(pi^2+1)`
/// (gp: each difference is 0). Before: `UndecidedZero` on the value rows, a value of a constant
/// factor on the `DivisionByZero` rows.
#[test]
fn m3r6_gaussian_constants_are_inverted_exactly() {
    let inv_x: &'static [Coef<'static>] = &[(-1, "1", "0")];
    let cases = vec![
        value("1/(x + 1/(1+I) - (1-I)/2)", "0", 2, inv_x, LimWant::Unknown),
        value(
            "1/(x + 1 + (9/8*I)/(1+I)^2 - 25/16)",
            "0",
            2,
            inv_x,
            LimWant::Unknown,
        ),
        value(
            "1/(x + 1/(1-2*I) - (1+2*I)/5)",
            "0",
            2,
            inv_x,
            LimWant::Unknown,
        ),
        value(
            "1/(x + 1/(pi+I) - (pi-I)/(pi^2+1))",
            "0",
            2,
            inv_x,
            LimWant::Unknown,
        ),
        refusal("x/(1/(1+I) - (1-I)/2)", "0", 2, &[Kind::DivisionByZero]),
        refusal("x/(3/(2-I) - (6+3*I)/5)", "0", 2, &[Kind::DivisionByZero]),
    ];
    run(cases);
}

/// m3: values through a Gaussian inverse (controls; these were values before too, with the
/// kernel 1/(A + B*I)), against gp.
#[test]
fn m3r6_values_through_gaussian_inverses() {
    // gp: 1/(x + 1/(2+3*I)) + O(x^3)
    let a: &'static [Coef<'static>] = &[(0, "2", "3"), (1, "5", "-12"), (2, "-46", "9")];
    // gp: 1/(x + 1/(Pi+I)) + O(x^3), \p 70
    let b: &'static [Coef<'static>] = &[
        (
            0,
            "3.141592653589793238462643383279502884197169399375105820974944592307816",
            "1",
        ),
        (
            1,
            "-8.869604401089358618834490999876151135313699407240790626413349376220045",
            "-6.283185307179586476925286766559005768394338798750211641949889184615633",
        ),
        (
            2,
            "21.58149871953044046008838491726288654963378036775979023121970432688295",
            "28.60881320326807585650347299962845340594109822172237187924004812866013",
        ),
    ];
    // gp: log(x + 1/(1+I)) + O(x^3), \p 70
    let c: &'static [Coef<'static>] = &[
        (
            0,
            "-0.3465735902799726547086160607290882840377500671801276270603400047466968",
            "-0.7853981633974483096156608458198757210492923498437764552437361480769541",
        ),
        (1, "1", "1"),
        (2, "0", "-1"),
    ];
    let cases = vec![
        value("1/(x + 1/(2+3*I))", "0", 2, a, LimWant::Unknown),
        value("1/(x + 1/(pi+I))", "0", 2, b, LimWant::Unknown),
        value("log(x + 1/(1+I))", "0", 2, c, LimWant::Unknown),
    ];
    run(cases);
}
