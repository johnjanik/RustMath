//! Complete-factorization drivers of `rustmath-integers` against PARI/GP `factor`.
//!
//! MAGMA Handbook ch. 18 §18.10 (digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`):
//! `Factorization(n)` returns the complete factorisation of |n| into primes. The three drivers
//! tested here play that role with one splitting method each:
//! - `prime::factor_pollard_rho`: Pollard rho (J. M. Pollard, *A Monte Carlo method for
//!   factorization*, BIT 15 (1975) 331-334) with Brent's cycle detection as the fallback
//!   (R. P. Brent, *An improved Monte Carlo factorization algorithm*, BIT 20 (1980) 176-184,
//!   [Bre80] in the digest);
//! - `quadratic_sieve::quadratic_sieve_factor_complete` (MAGMA `MPQS`);
//! - `ecm::ecm_factor_complete` (MAGMA `ECM`; H. W. Lenstra, *Factoring integers with
//!   elliptic curves*, Ann. of Math. 126 (1987) 649-673).
//!
//! Before X1b the QS driver pushed a cofactor its own loop guard had shown composite
//! ("Couldn't factor, assume it's prime", tracker blocker B-04), the QS and ECM drivers pushed
//! splitting divisors without checking that they were prime, and the ECM fallback pushed the
//! composite remainder a second time (`ecm_factor_complete(3127, 1000, 10000, 0)` gave
//! `[(53,1), (59,1), (3127,1)]`, whose product is 3127^2).
//!
//! Every check below asserts: each returned p passes `is_prime`, the product of the p^e is |n|,
//! and the list equals PARI/GP 2.15.4 `factor(n)` (primes ascending, PARI's order). The `gp`
//! command is in the comment next to each value.

use rustmath_integers::ecm::ecm_factor_complete;
use rustmath_integers::prime::{factor_pollard_rho, is_prime};
use rustmath_integers::quadratic_sieve::quadratic_sieve_factor_complete;
use rustmath_integers::Integer;

fn z(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap_or_else(|| panic!("bad integer {s:?}"))
}

/// One row: n and PARI/GP `factor(n)` as (prime, exponent) pairs in PARI's order.
type Case = (&'static str, &'static [(&'static str, u32)]);

/// Reference factorizations. The first six are the design's X1b list: every prime is above
/// 47, so the QS and ECM small-prime pre-division (primes <= 47) leaves the whole number to
/// the splitting method.
const CASES: &[Case] = &[
    // gp: factor(12780049) = [53,1; 59,1; 61,1; 67,1]
    ("12780049", &[("53", 1), ("59", 1), ("61", 1), ("67", 1)]),
    // gp: factor(1041537223) = [1009,1; 1013,1; 1019,1]
    ("1041537223", &[("1009", 1), ("1013", 1), ("1019", 1)]),
    // gp: factor(1031316053) = [1009,2; 1013,1]
    ("1031316053", &[("1009", 2), ("1013", 1)]),
    // gp: factor(13710311357) = [101,1; 103,1; 107,1; 109,1; 113,1]
    (
        "13710311357",
        &[("101", 1), ("103", 1), ("107", 1), ("109", 1), ("113", 1)],
    ),
    // gp: factor(25) = Mat([5,2])  (`prime::pollard_rho`, Floyd on x^2 + 1, returns None
    // for 25)
    ("25", &[("5", 2)]),
    // gp: factor(1225) = [5,2; 7,2]
    ("1225", &[("5", 2), ("7", 2)]),
    // gp: factor(3127) = [53,1; 59,1]; isprime(3127) = 0
    ("3127", &[("53", 1), ("59", 1)]),
    // gp: factor(2809) = Mat([53,2])
    ("2809", &[("53", 2)]),
    // gp: factor(165731) = [53,2; 59,1]
    ("165731", &[("53", 2), ("59", 1)]),
    // gp: factor(2^4*53^3*59^2) = [2,4; 53,3; 59,2]  (= 8291853392)
    ("8291853392", &[("2", 4), ("53", 3), ("59", 2)]),
    // gp: factor(2^2*3^3*1009^2*1013) = [2,2; 3,3; 1009,2; 1013,1]  (= 111382133724)
    (
        "111382133724",
        &[("2", 2), ("3", 3), ("1009", 2), ("1013", 1)],
    ),
    // gp: factor(117077) = [47,2; 53,1]
    ("117077", &[("47", 2), ("53", 1)]),
    // gp: factor(9381) = [3,1; 53,1; 59,1]
    ("9381", &[("3", 1), ("53", 1), ("59", 1)]),
    // gp: factor(15347) = [103,1; 149,1]
    ("15347", &[("103", 1), ("149", 1)]),
    // gp: factor(10403) = [101,1; 103,1]
    ("10403", &[("101", 1), ("103", 1)]),
    // gp: factor(123456789) = [3,2; 3607,1; 3803,1]
    ("123456789", &[("3", 2), ("3607", 1), ("3803", 1)]),
    // gp: factor(600851475143) = [71,1; 839,1; 1471,1; 6857,1]
    (
        "600851475143",
        &[("71", 1), ("839", 1), ("1471", 1), ("6857", 1)],
    ),
    // gp: factor(999962000357) = [999979,1; 999983,1]
    ("999962000357", &[("999979", 1), ("999983", 1)]),
    // gp: factor(2^32+1) = [641,1; 6700417,1]
    ("4294967297", &[("641", 1), ("6700417", 1)]),
    // gp: factor(2^64+1) = [274177,1; 67280421310721,1]
    (
        "18446744073709551617",
        &[("274177", 1), ("67280421310721", 1)],
    ),
    // gp: factor(1000003*1000033*1000037) = [1000003,1; 1000033,1; 1000037,1]
    (
        "1000073001431003663",
        &[("1000003", 1), ("1000033", 1), ("1000037", 1)],
    ),
    // gp: factor(6700417) = Mat([6700417,1]); isprime(6700417) = 1
    ("6700417", &[("6700417", 1)]),
];

/// Asserts that `got` is the PARI factorization of n: every prime passes `is_prime`, the
/// product of the p^e is |n|, and the list equals `expected`.
fn assert_factorization(
    who: &str,
    n: &Integer,
    got: &[(Integer, u32)],
    expected: &[(Integer, u32)],
) {
    for (p, e) in got {
        assert!(
            is_prime(p),
            "{who}({n}): factor {p} is not prime (got {got:?})"
        );
        assert!(*e >= 1, "{who}({n}): exponent 0 for {p} (got {got:?})");
    }
    let product = got
        .iter()
        .fold(Integer::one(), |acc, (p, e)| acc * p.pow(*e));
    assert_eq!(
        product,
        n.abs(),
        "{who}({n}): product of p^e is not |n| (got {got:?})"
    );
    assert_eq!(got, expected, "{who}({n}): differs from PARI factor");
}

fn expected(case: &Case) -> Vec<(Integer, u32)> {
    case.1.iter().map(|&(p, e)| (z(p), e)).collect()
}

fn run_all(who: &str, driver: impl Fn(&Integer) -> Vec<(Integer, u32)>) {
    for case in CASES {
        let n = z(case.0);
        let want = expected(case);
        assert_factorization(who, &n, &driver(&n), &want);
        // |n| is factored: the sign is dropped (PARI puts -1 as a factor; these drivers return
        // only the primes).
        let minus_n = -n.clone();
        assert_factorization(who, &minus_n, &driver(&minus_n), &want);
    }
}

#[test]
fn factor_pollard_rho_matches_pari() {
    run_all("factor_pollard_rho", factor_pollard_rho);
}

#[test]
fn quadratic_sieve_factor_complete_matches_pari() {
    run_all(
        "quadratic_sieve_factor_complete",
        quadratic_sieve_factor_complete,
    );
}

#[test]
fn ecm_factor_complete_matches_pari() {
    run_all("ecm_factor_complete(.., 1000, 10000, 10)", |n| {
        ecm_factor_complete(n, 1000, 10000, 10)
    });
}

/// `curves = 0` is accepted and makes `ecm_factor` return `None` for every odd composite
/// without a prime factor <= 47, so this runs the fallback on every such row.
#[test]
fn ecm_factor_complete_zero_curves_matches_pari() {
    run_all("ecm_factor_complete(.., 1000, 10000, 0)", |n| {
        ecm_factor_complete(n, 1000, 10000, 0)
    });
}

/// Review R4: the fallback appended the complete factorization of the remainder and then
/// pushed the composite remainder again.
#[test]
fn ecm_fallback_does_not_push_the_remainder_twice() {
    // gp: factor(3127) = [53,1; 59,1]; isprime(3127) = 0
    let got = ecm_factor_complete(&Integer::from(3127), 1000, 10000, 0);
    assert_eq!(got, vec![(Integer::from(53), 1), (Integer::from(59), 1)]);
}

/// Trivial inputs keep their documented values: no factors for 0, 1 and -1.
#[test]
fn trivial_inputs() {
    for n in [0i64, 1, -1] {
        let n = Integer::from(n);
        assert!(factor_pollard_rho(&n).is_empty(), "factor_pollard_rho({n})");
        assert!(quadratic_sieve_factor_complete(&n).is_empty(), "qs({n})");
        assert!(
            ecm_factor_complete(&n, 1000, 10000, 0).is_empty(),
            "ecm({n})"
        );
    }
}

/// psi_12 = 318665857834031151167461 passes the twelve strong tests 2..37, so the fixed-base
/// test accepted it and every driver returned it unsplit. With the proving `is_prime`
/// (Baillie-PSW rejects it) the loop guard splits it. gp: factor(318665857834031151167461) =
/// [399165290221, 1; 798330580441, 1]. About 5 s per driver in a debug build (Floyd's rho needs
/// 188311 steps), so the three drivers are three tests.
fn psi12_factorization() -> (Integer, Vec<(Integer, u32)>) {
    (
        z("318665857834031151167461"),
        vec![(z("399165290221"), 1), (z("798330580441"), 1)],
    )
}

#[test]
fn psi12_is_split_by_factor_pollard_rho() {
    let (n, want) = psi12_factorization();
    assert_factorization("factor_pollard_rho", &n, &factor_pollard_rho(&n), &want);
}

#[test]
fn psi12_is_split_by_quadratic_sieve_factor_complete() {
    let (n, want) = psi12_factorization();
    let got = quadratic_sieve_factor_complete(&n);
    assert_factorization("quadratic_sieve_factor_complete", &n, &got, &want);
}

#[test]
fn psi12_is_split_by_ecm_factor_complete() {
    let (n, want) = psi12_factorization();
    let got = ecm_factor_complete(&n, 1000, 10000, 0);
    assert_factorization("ecm_factor_complete(.., 0)", &n, &got, &want);
}
