//! S1: a fixed sample of review 9's random harness, with PARI/GP references.
//!
//! `data/s1_fuzz.tsv` holds the first 60 inputs of each of review 9's eight random runs
//! (`EXP/review9/fuzz.py` with seeds 1 to 4 and `fuzz2.py`, which adds the inverse functions, with
//! seeds 11 to 14; EXP = the S1 design's prototype folder): random expressions of depth 2 to 4 in
//! the functions of design §2.4 (abs, sign and the inverse functions included), sums, products,
//! quotients, integer, rational, signed and series exponents, constant bases to series, and exact
//! constants in the variable (`x - x`, `(x+1)^2-x^2-2*x-1`, `0*((1)/(x-a))`, `(x-a)/(x-a)`,
//! `sign(x^2+1)`), at the points 0, 1, -1/2, 2, 1/3, -1 and orders 0 to 3. The generator,
//! `data/gen_s1_fuzz.py` (`python3 -I gen_s1_fuzz.py > s1_fuzz.tsv` in `tests/data`), replays
//! review 9's random sequences (checked against its output files), computes gp's series at
//! realprecision 300 (`data/s1_gp.py`) and the prototype's outcome, and classes each row from the
//! two outcomes (it does not compare the prototype's value with gp's; the tests do that):
//!
//! - `V` (value): gp gives a series and the prototype a value; gp's series, compared coefficient
//!   by coefficient to 60 digits.
//! - `R` (refusal): gp fails and the prototype refuses; the prototype's kind, reviewed by hand
//!   against the design and the owner's rules. Four rows also accept a second refusal kind (` | `
//!   in the data): `R:y11.25` and `R:z2.56`, where the frozen design's constant field cannot see
//!   that the constant term -pi*exp(-log(pi)) is -1 (true outcome `NotLaurent`) or that
//!   sin(1)^2 + cos(1)^2 is 1 (true outcome `DivisionByZero`: the input raises acos(1) = 0 to the
//!   power -1); `R:y11.42` and `R:y13.13`, which have two refusal causes, so the kind follows the
//!   evaluation order (`UndecidedZero` or `NotLaurent`; `Unsupported` or `NotLaurent`).
//! - `U` (undecided): gp decides with floating point, the design's exact arithmetic cannot
//!   (`UndecidedZero` of a complex kernel without an enclosure, of a non-constant operand;
//!   `Unsupported` abs or sign of a non-constant series with non-rational coefficients); the kind
//!   is asserted. The 11 `UndecidedZero` rows also accept gp's value: the undecided constant is
//!   nonzero, so a port that decides it gives gp's series (each value checked against gp's
//!   `derivnum` of the scalar function; `U:y12.1` is on the cut of acos, where the value is the
//!   branch whose value at the point is principal, design §2.4 "Branch at the point").
//! - `B1`, `B2`, `N`: rows of review 9's blockers and gp's formal branches, reviewed by hand (the
//!   reason is in the note column): an exact-constant argument of the atan family is evaluated (B1,
//!   a value against gp; this includes an argument whose integrand factor the prototype could not
//!   decide, `B1:y14.33` atan(atanh(3)) and `B1:y13.19`, since the owner's rule forms no integrand),
//!   a non-constant argument at arctanh's poles is `NotLaurent` (B2), and a square-root branch
//!   point of valuation 2 is `NotLaurent` where gp gives a formal one-sided series (N).
//! - `O`: rows whose outcome depends on a fold that the scope of the owner's rule leaves optional
//!   (`s1_common/mod.rs`), with every honest outcome accepted (` | ` in the data): sign of an exact
//!   non-rational constant (`Unsupported` or the value), exp or tan of the exact 0 (the exact
//!   constant, or a truncated one and then `PrecisionExhausted`).
//! - `E`: gp fails on an exact-zero shape (an exact zero raised to a series, a series without
//!   terms) and the prototype gives a value; 7 rows reviewed by hand, each against gp's series of
//!   the simplified input (in the note column). The eighth such input of the sample, `z1.9`
//!   (log(log(sin(1)^2 + cos(1)^2)), a kernel at an undecided pole, design §7 item 10), is left out.
//!
//! The limit column (§2.13: `Finite` only for a rational constant term) accepts `Finite(q)` or
//! `Unknown` where gp's constant term is the rational q but the design's constant field keeps a
//! kernel such as exp(4 log 2) (`V:z2.57`, 2^(2^x) at 2).
//!
//! Each row goes through `series_expansion` and `taylor_with_budget` with the REPL budget and the
//! limit path, on a thread with the bar's stack (2 MB in debug, 8 MB in release). The prototype of
//! the design passes every row except the B1 and B2 rows and `O:y13.36` (whose asin(1) is
//! proto13's B1 `DivisionByZero`); review 9 found about 20 and 25 of its 2000 inputs in those
//! classes.
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`);
//! Knuth, TAOCP vol. 2, §4.7; Brent and Kung, J. ACM 25(4), 1978, 581-595.

mod s1_common;

use s1_common::check::{check_all, parse_rows, Row, RowWant};

const DATA: &str = include_str!("data/s1_fuzz.tsv");
const DIGITS: u32 = 60;
const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

fn rows(class: &str) -> Vec<Row> {
    parse_rows(DATA)
        .into_iter()
        .filter(|r| r.id.split(':').next() == Some(class))
        .collect()
}

#[test]
fn data_has_at_least_300_rows_with_gp_references() {
    let all = parse_rows(DATA);
    assert!(all.len() >= 300, "{} rows", all.len());
    let values = all
        .iter()
        .filter(|r| matches!(r.want, RowWant::Value(_)))
        .count();
    assert!(values >= 300, "{values} value rows");
    for r in &all {
        assert!(
            r.note.contains("gp"),
            "row {} has no gp outcome in its note",
            r.id
        );
    }
}

/// Class V: the engine's coefficients equal gp's series to 60 digits.
#[test]
fn values_match_gp() {
    check_all(rows("V"), DIGITS, STACK);
}

/// Class R: gp fails, the engine refuses with the design's kind (or, on four rows, the second
/// honest kind the row lists).
#[test]
fn refusals_where_gp_fails() {
    check_all(rows("R"), DIGITS, STACK);
}

/// Class U: the design's exact arithmetic cannot decide what gp decides in floating point (an
/// `UndecidedZero` row also accepts gp's value).
#[test]
fn undecided_where_gp_uses_floats() {
    check_all(rows("U"), DIGITS, STACK);
}

/// Classes B1, B2 and N: review 9's blockers and gp's formal branches, reviewed by hand.
#[test]
fn hand_reviewed_blocker_rows() {
    let mut r = rows("B1");
    r.extend(rows("B2"));
    r.extend(rows("N"));
    assert!(r.len() >= 10, "{} hand-reviewed rows", r.len());
    check_all(r, DIGITS, STACK);
}

/// Class E: values where gp fails on the exact-zero shape, against gp on the simplified input.
#[test]
fn values_where_gp_fails_on_the_shape() {
    let r = rows("E");
    assert!(r.len() >= 7, "{} rows of class E", r.len());
    check_all(r, DIGITS, STACK);
}

/// Class O: an optional fold of the owner's rule; one of the listed outcomes, nothing else.
#[test]
fn rows_with_an_optional_fold() {
    let r = rows("O");
    assert!(r.len() >= 4, "{} rows of class O", r.len());
    for row in &r {
        assert!(
            matches!(row.want, RowWant::OneOf(_)),
            "row {} of class O lists one outcome",
            row.id
        );
    }
    check_all(r, DIGITS, STACK);
}
