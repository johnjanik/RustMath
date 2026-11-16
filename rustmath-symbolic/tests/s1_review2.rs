//! S1 implementation, fix-up round 2: the minor items of review 2 (semantics lens,
//! `fu2/s1impl/S1.review2-semantics.md`) that this round fixes, as regression tests.
//!
//! - Minor 4: `sin_q_pi` read 12 q with `to_i64`, so sin, cos and tan at q pi were exact only for
//!   |12 q| < 2^63: `sin(10^20*pi)` and `cos(10^20*pi) - 1` were `UndecidedZero` (both are 0), and
//!   `x*tan((2^60+1/2)*pi)` was `Ok` with the kernel tan(..) and the limit `Finite(0)`, while
//!   `x*tan((2^58+1/2)*pi)` was `DivisionByZero` (tan has a pole at every odd multiple of pi/2).
//!   Now 12 q is reduced mod 24 as an Integer.
//! - Minor 2: exp of a sum of integer multiples of logs of rationals stayed a kernel, so `x^x` and
//!   `3^x` at 3 had the constant term exp(3 log(3)) and the limit path gave `Unknown` where the limit
//!   is 27, and `1/(2^x - 8)` at 3 was `UndecidedZero` where it has a pole. Now
//!   exp(r + n_1 log(q_1) + ...) is exp(r) q_1^n_1 ... (the inverse of `log_exact`).
//!
//! References: PARI/GP 2.15.4 at realprecision 300 (`fu2/s1impl/fix2/r2.gp`, output `r2_gp.txt`),
//! with the gp command in a comment; 70 digits written, 60 compared. gp's `Pi` is a float, so
//! `sin(10^20*Pi)` is -4.2e-289 there, not 0; the exact values are sin(n pi) = 0, sin(pi/6) =
//! cos(pi/3) = 1/2, tan(pi/4) = 1, and cos(pi/2) = 0 (a pole of tan).
//!
//! MAGMA Handbook ch. 49 (digest `rustmath-lava/docs/Chapter_49_Power_Laurent_and_Puiseux_Series.md`,
//! §49.5 transcendental functions of a series); S1 design §2.2 (the constant field: exact values of
//! sin, cos, tan at rational multiples of pi; log of a product).

mod s1_common;

use s1_common::check::*;

const DIGITS: u32 = 60;
const STACK: usize = if cfg!(debug_assertions) {
    2 << 20
} else {
    8 << 20
};

fn value(
    f: &'static str,
    a: &'static str,
    order: i64,
    want: &'static [Coef<'static>],
    lim: &'static str,
) {
    on_stack(STACK, move || {
        check_value(f, a, order, want, &lim_want_from(lim), DIGITS)
            .unwrap_or_else(|m| panic!("{f} at {a}, order {order}: {m}"));
    });
}

/// Minor 4: sin, cos and tan at q pi for |12 q| >= 2^63 are exact.
#[test]
fn m4_trig_at_large_rational_multiples_of_pi_is_exact() {
    // sin(n pi) = 0 and cos(n pi) - 1 = 0 for the even n = 10^20 (gp: `sin(10^20*Pi)` is
    // -4.2e-289 and `cos(10^20*Pi)-1` is -8.7e-578, the float error of Pi)
    value("sin(10^20*pi) + x", "0", 1, &[(1, "1", "0")], "Finite(0)");
    value(
        "cos(10^20*pi) - 1 + x",
        "0",
        1,
        &[(1, "1", "0")],
        "Finite(0)",
    );
    // gp: `sin((10^20+1/6)*Pi)` 0.5, `cos((2^70+1/3)*Pi)` 0.5, `tan((10^30+1/4)*Pi)` 1.0
    value(
        "sin((10^20+1/6)*pi) + x",
        "0",
        1,
        &[(0, "0.5", "0"), (1, "1", "0")],
        "Finite(1/2)",
    );
    value(
        "cos((2^70+1/3)*pi) + x",
        "0",
        1,
        &[(0, "0.5", "0"), (1, "1", "0")],
        "Finite(1/2)",
    );
    value(
        "tan((10^30+1/4)*pi) + x",
        "0",
        1,
        &[(0, "1", "0"), (1, "1", "0")],
        "Finite(1)",
    );
    // tan at an odd multiple of pi/2 is a pole of the constant: DivisionByZero for 2^58 (as
    // before) and for 2^60 (before: Ok with the kernel and the limit Finite(0)). gp:
    // `cos((2^60+1/2)*Pi)` is 4.8e-291, the float error of Pi.
    for f in [
        "x*tan((2^58+1/2)*pi)",
        "x*tan((2^60+1/2)*pi)",
        "x*tan((10^25+1/2)*pi)",
    ] {
        on_stack(STACK, move || {
            check_refusal(f, "0", 2, Kind::DivisionByZero, &LimWant::Unknown)
                .unwrap_or_else(|m| panic!("{f} at 0: {m}"));
        });
    }
}

/// Minor 2: exp of integer multiples of logs of rationals is the rational product.
#[test]
fn m2_exp_of_integer_multiples_of_logs_is_rational() {
    // gp: `exp((3+y)*log(3+y))+O(y^3)`: 27, 56.66.. (27 + 27 log 3), 63.95.. (18 + 27 log 3 +
    // 27/2 log(3)^2)
    value(
        "x^x",
        "3",
        2,
        &[
            (0, "27", "0"),
            (
                1,
                "56.662531794038961667671621396908194025482245061214235196836747008212345916902",
                "0",
            ),
            (
                2,
                "63.956342765008818368562639568874633859331500240708731991862559348998255366810",
                "0",
            ),
        ],
        "Finite(27)",
    );
    // gp: `exp((3+y)*log(3))+O(y^3)`
    value(
        "3^x",
        "3",
        2,
        &[
            (0, "27", "0"),
            (
                1,
                "29.662531794038961667671621396908194025482245061214235196836747008212345916903",
                "0",
            ),
            (
                2,
                "16.293810970969856700891018171966439833849255179494496795025812340785909449908",
                "0",
            ),
        ],
        "Finite(27)",
    );
    // gp: `exp((3+y)*log(2))+O(y^2)`
    value(
        "2^x",
        "3",
        1,
        &[
            (0, "8", "0"),
            (
                1,
                "5.5451774444795624753378569716654125446040010748820420329654400759471489757576",
                "0",
            ),
        ],
        "Finite(8)",
    );
    // the pole of 1/(2^x - 8) at 3 (the constant term of 2^x - 8 is now the exact 0). gp:
    // `1/(8*(exp(y*log(2))-1))+O(y^2)`: 1/(8 log 2) y^-1 - 1/16 + log(2)/96 y
    value(
        "1/(2^x - 8)",
        "3",
        1,
        &[
            (
                -1,
                "0.18033688011112042591999058512523651717833074426912324176693117586638865239765",
                "0",
            ),
            (0, "-0.0625", "0"),
            (
                1,
                "0.0072202831308327636397628345985226725841197930662526588970904167655561835621843",
                "0",
            ),
        ],
        "Unknown",
    );
    // a negative rational: exp((x + 1) log(-2)) = -2 exp(x log(-2)), principal log(-2) =
    // log 2 + i pi. gp: `-2*exp(y*log(-2))+O(y^2)`
    value(
        "exp((x + 1)*log(-2))",
        "0",
        1,
        &[
            (0, "-2", "0"),
            (
                1,
                "-1.38629436111989061883446424291635313615100026872051050824136001898678724393939",
                "-6.2831853071795864769252867665590057683943387987502116419498891846156328125724",
            ),
        ],
        "Finite(-2)",
    );
    // with a rational part: exp(x + 2 log 3) = 9 exp(x). gp: `exp(y+2*log(3))+O(y^2)`: 9, 9
    value(
        "exp(x + 2*log(3))",
        "0",
        1,
        &[(0, "9", "0"), (1, "9", "0")],
        "Finite(9)",
    );
    // a product larger than MAX_EXACT_POWER_BITS stays the kernel exp(10^6 log 3): a value,
    // fast, and the limit path does not read it as rational
    on_stack(STACK, || {
        let f = s1_common::api::parse("x + exp(1000000*log(3))");
        let s = s1_common::api::se(
            &f,
            &x(),
            &s1_common::api::ExpansionPoint::At(s1_common::api::parse("0")),
            1,
            Some(s1_common::api::REPL_TAYLOR_WORK),
        )
        .unwrap_or_else(|e| panic!("x + exp(10^6 log 3): {e:?}"));
        assert_eq!(s.terms.len(), 2);
        assert!(
            tree_size(&s.terms[0].1) < 64,
            "the constant term is a small kernel"
        );
        assert_eq!(
            s1_common::api::lim(&f, &x(), &s1_common::api::parse("0")),
            s1_common::api::LimitResult::Unknown
        );
    });
}
