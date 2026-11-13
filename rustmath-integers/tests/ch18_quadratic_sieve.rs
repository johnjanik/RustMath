//! The self-initialising quadratic sieve against PARI/GP.
//!
//! MAGMA Handbook ch. 18 §18.10.3 `MPQS` (digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`);
//! S. P. Contini, *Factoring integers with the self-initializing quadratic sieve*, MSc thesis,
//! University of Georgia, 1997; R. Crandall, C. Pomerance, *Prime Numbers: A Computational
//! Perspective*, 2nd ed., Springer 2005, §6.1.
//!
//! The semiprimes of d digits are reproduced by (PARI/GP 2.15.4)
//!
//! ```text
//! default(realprecision, 120); h = d\2; p = nextprime(floor(Pi*10^(h-1)));
//! q = nextprime(floor(sqrt(10)*exp(1)*10^(d-h-1))); [p, q, p*q, factor(p*q) == [p,1; q,1]]
//! ```
//!
//! Before this unit the sieve returned `None` on every semiprime tried (10403, 20 to 30
//! digits): its GF(2) step returned sets that were not dependencies and its square root was a
//! floor root of a non-square.

use rustmath_integers::quadratic_sieve::{
    quadratic_sieve_factor, quadratic_sieve_factor_complete, quadratic_sieve_factor_with_params,
    QsOutcome, QsParams,
};
use rustmath_integers::Integer;
use std::sync::mpsc;
use std::time::Duration;

fn z(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap_or_else(|| panic!("bad integer {s:?}"))
}

/// Run `f` on a thread and fail if it does not finish within `secs` seconds. A panic inside
/// `f` (a failed assertion) is re-raised here with its own message, not reported as a timeout.
/// (A copy of the helper in `rustmath-polynomials/tests/ch23_quotrem_gcd.rs`.)
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(v) => v,
        // The sender was dropped without sending: `f` panicked.
        Err(mpsc::RecvTimeoutError::Disconnected) => match worker.join() {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(()) => unreachable!("the worker returned without sending its result"),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("timed out after {secs} s: a non-terminating or blown-up computation is back")
        }
    }
}

/// (d, n, p, q) from the gp command of the module doc.
const ROWS: &[(u32, &str, &str, &str)] = &[
    (20, "27005010885249061781", "3141592661", "8595961921"),
    (
        25,
        "2700501075624869745621467",
        "314159265359",
        "8595961900213",
    ),
    (
        30,
        "270050107561429816771648674403",
        "314159265359057",
        "859596190017779",
    ),
    (
        35,
        "27005010756136059108789661003408639",
        "31415926535897999",
        "859596190017769361",
    ),
    (
        40,
        "2700501075613600140300554153516920373243",
        "31415926535897932429",
        "85959619001776934567",
    ),
    (
        45,
        "270050107561360013478271527386296955458256719",
        "3141592653589793238499",
        "85959619001776934511781",
    ),
    (
        50,
        "27005010756136001347490384749379527844355732904871",
        "3141592653589793238462773",
        "8595961900177693451170027",
    ),
    (
        55,
        "2700501075613600134748901272440129406658374730908723747",
        "314159265358979323846264367",
        "8595961900177693451169944141",
    ),
    (
        60,
        "270050107561360013474890101164188225207220555802656708135689",
        "314159265358979323846264338521",
        "859596190017769345116994409009",
    ),
];

/// gp: nextprime(10^74) * nextprime(3*10^75); #binary(%) = 497.
const N150: &str = "300000000000000000000000000000000000000000000000000000000000000000000000627700000000000000000000000000000000000000000000000000000000000000000000013869";
/// gp: nextprime(10^29)
const P30: &str = "100000000000000000000000000319";

fn check_rows(lo: u32, hi: u32) {
    for &(d, n, p, q) in ROWS.iter().filter(|r| (lo..=hi).contains(&r.0)) {
        let (n, p, q) = (z(n), z(p), z(q));
        let d_found = quadratic_sieve_factor(&n).unwrap_or_else(|| panic!("{d} digits: None"));
        assert!(d_found == p || d_found == q, "{d} digits: {d_found}");
        if d == 20 || d == 30 {
            let d_found = quadratic_sieve_factor(&-n.clone()).expect("negative n");
            assert!(d_found == p || d_found == q, "-n, {d} digits: {d_found}");
        }
    }
}

#[test]
fn qs_factors_gp_semiprimes_20_to_45_digits() {
    check_rows(20, 45);
}

#[test]
#[ignore = "about 30 s in a debug build; run with `cargo test --release -p rustmath-integers --test ch18_quadratic_sieve -- --ignored` (about 4 s)"]
fn qs_factors_gp_semiprimes_50_to_60_digits() {
    check_rows(50, 60);
}

fn assert_proper_divisor(d: &Integer, n: &Integer) {
    assert!(
        *d > Integer::one() && d < n && (n % d).is_zero(),
        "{d} is not a proper divisor of {n}"
    );
}

#[test]
fn qs_factors_named_inputs() {
    for n in [
        // gp: factor(2^128+1) = [59649589127497217, 1; 5704689200685129054721, 1]
        Integer::from(2).pow(128) + Integer::one(),
        // gp: factor(100000190000000000000090000171) = [10000019, 1; 10000000000000000000009, 1]
        z("100000190000000000000090000171"),
        // gp: factor(6000000113000000706000001463) =
        //     [1000000007, 1; 2000000011, 1; 3000000019, 1]
        z("6000000113000000706000001463"),
        // gp: factor(100000014000001100000085400002989) = [100000007, 2; 10000000000000061, 1]
        z("100000014000001100000085400002989"),
    ] {
        let d = quadratic_sieve_factor(&n).unwrap_or_else(|| panic!("{n}: None"));
        assert_proper_divisor(&d, &n);
    }
}

fn pairs(v: &[(&str, u32)]) -> Vec<(Integer, u32)> {
    v.iter().map(|&(p, e)| (z(p), e)).collect()
}

#[test]
fn qs_complete_matches_pari() {
    let cases: Vec<(Integer, Vec<(Integer, u32)>)> = vec![
        (
            z(ROWS[4].1),
            pairs(&[("31415926535897932429", 1), ("85959619001776934567", 1)]),
        ),
        (
            Integer::from(2).pow(128) + Integer::one(),
            pairs(&[("59649589127497217", 1), ("5704689200685129054721", 1)]),
        ),
        (
            -z("6000000113000000706000001463"),
            pairs(&[("1000000007", 1), ("2000000011", 1), ("3000000019", 1)]),
        ),
        (
            z("100000014000001100000085400002989"),
            pairs(&[("100000007", 2), ("10000000000000061", 1)]),
        ),
        // psi_12 passes the twelve strong tests 2..37; Baillie-PSW rejects it.
        // gp: factor(318665857834031151167461) = [399165290221, 1; 798330580441, 1]
        (
            z("318665857834031151167461"),
            pairs(&[("399165290221", 1), ("798330580441", 1)]),
        ),
    ];
    for (n, want) in cases {
        assert_eq!(quadratic_sieve_factor_complete(&n), want, "n = {n}");
    }
}

/// The driver splits a prime power with one perfect-power test, not one split per exponent.
#[test]
fn qs_complete_prime_powers() {
    // gp: factor(3^2000) = Mat([3, 2000]); factor(1000003^50) = Mat([1000003, 50]);
    //     factor(2^1000*3^500) = [2, 1000; 3, 500]
    let got = within(5, || {
        [
            Integer::from(3).pow(2000),
            Integer::from(1000003).pow(50),
            Integer::from(2).pow(1000) * Integer::from(3).pow(500),
        ]
        .map(|n| quadratic_sieve_factor_complete(&n))
    });
    assert_eq!(got[0], vec![(Integer::from(3), 2000)]);
    assert_eq!(got[1], vec![(Integer::from(1000003), 50)]);
    assert_eq!(
        got[2],
        vec![(Integer::from(2), 1000), (Integer::from(3), 500)]
    );
}

#[test]
fn qs_prime_gives_no_factor() {
    // gp: isprime(nextprime(10^29)) = 1: the probable-prime check returns None at once.
    assert_eq!(quadratic_sieve_factor(&z(P30)), None);
    // gp: isprime(nextprime(10^19)) = 1; nextprime(10^19) = 10000000000000000051. With
    // explicit parameters the sieve runs, and every dependency gives gcd 1 or n.
    let p20 = z("10000000000000000051");
    match quadratic_sieve_factor_with_params(&p20, &QsParams::for_bits(64)) {
        Ok(QsOutcome::NoFactor { dependencies }) => assert!(dependencies >= 32, "{dependencies}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn qs_with_params_tiny_inputs() {
    let params = QsParams::for_bits(12);
    // gp: primes(15) = [2, 3, 5, ..., 47]: +-1 and the primes up to 47 have no proper divisor.
    for n in [1i64, -1, 2, -2, 3, 5, 43, 47, -47] {
        assert_eq!(
            quadratic_sieve_factor_with_params(&Integer::from(n), &params),
            Ok(QsOutcome::NoFactor { dependencies: 0 }),
            "n = {n}"
        );
    }
    // Step 0 order: 2, then the odd primes up to 47. gp: factor(2209) = Mat([47, 2])
    for (n, d) in [
        (4i64, 2),
        (9, 3),
        (15, 3),
        (25, 5),
        (35, 5),
        (45, 3),
        (46, 2),
        (49, 7),
        (2209, 47),
    ] {
        assert_eq!(
            quadratic_sieve_factor_with_params(&Integer::from(n), &params),
            Ok(QsOutcome::Factor(Integer::from(d))),
            "n = {n}"
        );
    }
    // gp: isprime(2207) = 1. A prime above 47 runs the sieve.
    match quadratic_sieve_factor_with_params(&Integer::from(2207), &params) {
        Ok(QsOutcome::NoFactor { dependencies }) => assert!(dependencies >= 32, "{dependencies}"),
        other => panic!("{other:?}"),
    }
    assert!(quadratic_sieve_factor_with_params(&Integer::zero(), &params).is_err());
}

#[test]
fn qs_quick_paths() {
    let qs = |n: Integer| quadratic_sieve_factor(&n);
    assert_eq!(qs(Integer::from(100)), Some(Integer::from(2)));
    assert_eq!(qs(Integer::from(3) * z(P30)), Some(Integer::from(3)));
    // gp: factor(1000003^3) = Mat([1000003, 3])
    assert_eq!(
        qs(Integer::from(1000003).pow(3)),
        Some(Integer::from(1000003))
    );
    // gp: factor(10403) = [101, 1; 103, 1]: the factor base scan meets 101.
    assert_eq!(qs(Integer::from(10403)), Some(Integer::from(101)));
    // gp: factor(15) = [3, 1; 5, 1]
    assert_eq!(qs(Integer::from(-15)), Some(Integer::from(3)));
    for n in [0i64, 1, -1, 2, 3, 47] {
        assert_eq!(qs(Integer::from(n)), None, "n = {n}");
    }
}

#[test]
fn qs_out_of_polynomials() {
    let n = z(ROWS[4].1);
    let params = QsParams {
        max_polynomials: 1,
        ..QsParams::for_bits(n.bits())
    };
    match quadratic_sieve_factor_with_params(&n, &params) {
        Ok(QsOutcome::OutOfPolynomials { relations, needed }) => {
            assert!(relations < needed, "{relations} >= {needed}")
        }
        other => panic!("{other:?}"),
    }
}

/// N150 with a factor base of 30 or 31 primes needs s = 34 a-factors, but only 22 primes of the
/// base are above 30: the a-supply guard answers at once instead of drawing forever.
#[test]
fn qs_a_supply_too_small_returns_promptly() {
    let n = z(N150);
    let got = within(5, move || {
        [30usize, 31].map(|f| {
            let params = QsParams {
                factor_base_size: f,
                ..QsParams::for_bits(n.bits())
            };
            quadratic_sieve_factor_with_params(&n, &params)
        })
    });
    assert_eq!(
        got[0],
        Ok(QsOutcome::OutOfPolynomials {
            relations: 0,
            needed: 63
        })
    );
    assert_eq!(
        got[1],
        Ok(QsOutcome::OutOfPolynomials {
            relations: 0,
            needed: 64
        })
    );
}

/// The polynomial budget is checked before every polynomial, not once per a-value: with
/// F = 40 the guard passes with s = 31, and the first a has 2^30 polynomials.
#[test]
fn qs_polynomial_budget_is_per_polynomial() {
    let n = z(N150);
    let got = within(5, move || {
        [40usize, 60, 100].map(|f| {
            let params = QsParams {
                factor_base_size: f,
                max_polynomials: 10,
                ..QsParams::for_bits(497)
            };
            quadratic_sieve_factor_with_params(&n, &params)
        })
    });
    for (outcome, needed) in got.into_iter().zip([73, 93, 133]) {
        assert_eq!(
            outcome,
            Ok(QsOutcome::OutOfPolynomials {
                relations: 0,
                needed
            })
        );
    }
}

#[test]
fn qs_params_validation() {
    let base = QsParams::for_bits(64);
    let bad: Vec<(QsParams, &str)> = vec![
        (
            QsParams {
                factor_base_size: 29,
                ..base.clone()
            },
            "factor_base_size",
        ),
        (
            QsParams {
                factor_base_size: 65537,
                ..base.clone()
            },
            "factor_base_size",
        ),
        (
            QsParams {
                sieve_half_width: 63,
                ..base.clone()
            },
            "sieve_half_width",
        ),
        (
            QsParams {
                sieve_half_width: (1 << 24) + 1,
                ..base.clone()
            },
            "sieve_half_width",
        ),
        (
            QsParams {
                extra_relations: 0,
                ..base.clone()
            },
            "extra_relations",
        ),
        (
            QsParams {
                extra_relations: 1025,
                ..base.clone()
            },
            "extra_relations",
        ),
        (
            QsParams {
                large_prime_multiplier: 0,
                ..base.clone()
            },
            "large_prime_multiplier",
        ),
        (
            QsParams {
                max_polynomials: 0,
                ..base.clone()
            },
            "max_polynomials",
        ),
    ];
    for (params, field) in &bad {
        let err = params.validate().expect_err(field);
        assert!(format!("{err}").contains(field), "{err}");
    }
    let params = QsParams {
        factor_base_size: 65537,
        ..base.clone()
    };
    assert!(quadratic_sieve_factor_with_params(&Integer::from(10403), &params).is_err());
    let mut last = 0;
    for b in [0u64, 1, 2, 40, 64, 200, 332, 400, 2000, 1_000_000, u64::MAX] {
        let p = QsParams::for_bits(b);
        assert_eq!(p.validate(), Ok(()), "bits = {b}");
        assert!(p.factor_base_size >= last, "bits = {b}");
        last = p.factor_base_size;
    }
    assert_eq!(QsParams::for_bits(u64::MAX).factor_base_size, 65536);
    assert_eq!(QsParams::for_bits(0).factor_base_size, 60);
}

/// |y| de-duplication is load-bearing on this input: without it, the first prototype ended
/// with `None` after 42 dependencies with seed 4.
#[test]
fn qs_seed_sweep_24_digits() {
    // gp: factor(666958488061768516158211) = [765944099509, 1; 870766533079, 1]
    let n = z("666958488061768516158211");
    let (p, q) = (z("765944099509"), z("870766533079"));
    for seed in 1..=6 {
        let params = QsParams {
            seed,
            ..QsParams::for_bits(80)
        };
        match quadratic_sieve_factor_with_params(&n, &params) {
            Ok(QsOutcome::Factor(d)) => assert!(d == p || d == q, "seed {seed}: {d}"),
            other => panic!("seed {seed}: {other:?}"),
        }
    }
}
