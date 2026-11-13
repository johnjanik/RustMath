//! Factorization under limits against PARI/GP: MAGMA's third return value.
//!
//! MAGMA Handbook ch. 18 §18.10.1 (`Factorization(n : TrialDivisionLimit, PollardRhoLimit,
//! MPQSLimit, Proof)` and the sequence of unfactored composites) and §18.10.3
//! (`PollardRho(n, c, s, k)`, `TrialDivision`); digest
//! `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`. R. P. Brent, *An improved Monte Carlo
//! factorization algorithm*, BIT 20 (1980) 176-184; E. Bach, J. Driscoll, J. Shallit, *Factor
//! refinement*, J. Algorithms 15 (1993) 199-222.
//!
//! Every result is checked against the invariants of `LimitedFactorization`: value() == n, the
//! bases of the three lists pairwise coprime, each list sorted, `primes` entries pass
//! `is_prime`, `composites` entries fail `is_probable_prime`. Reference values: PARI/GP 2.15.4,
//! the command next to each value.

use rustmath_integers::factorization::{
    factorization_with_limits, FactorizationLimits, LimitedFactorization,
};
use rustmath_integers::prime::{
    factor_pollard_rho_limited, is_prime, is_probable_prime, POLLARD_RHO_DEFAULT_ITERATIONS,
};
use rustmath_integers::Integer;
use std::sync::mpsc;
use std::time::Duration;

fn z(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap_or_else(|| panic!("bad integer {s:?}"))
}

fn pairs(v: &[(&str, u32)]) -> Vec<(Integer, u32)> {
    v.iter().map(|&(p, e)| (z(p), e)).collect()
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

/// The invariants of `LimitedFactorization` for the input n.
fn check(n: &Integer, f: &LimitedFactorization) {
    assert_eq!(f.value(), *n, "value() != n: {f:?}");
    assert_eq!(f.sign, n.signum(), "{f:?}");
    let all: Vec<&(Integer, u32)> = f
        .primes
        .iter()
        .chain(&f.probable_primes)
        .chain(&f.composites)
        .collect();
    for (i, (a, e)) in all.iter().enumerate() {
        assert!(*a > Integer::one() && *e >= 1, "{f:?}");
        for (b, _) in &all[i + 1..] {
            assert!(a.gcd(b).is_one(), "{a} and {b} share a factor: {f:?}");
        }
    }
    for list in [&f.primes, &f.probable_primes, &f.composites] {
        assert!(list.windows(2).all(|w| w[0].0 < w[1].0), "unsorted: {f:?}");
    }
    for (p, _) in &f.primes {
        assert!(is_prime(p), "{p} is not prime: {f:?}");
    }
    for (q, _) in &f.probable_primes {
        assert!(is_probable_prime(q), "{q} is not a probable prime: {f:?}");
    }
    for (c, _) in &f.composites {
        assert!(!is_probable_prime(c), "{c} is not composite: {f:?}");
    }
}

fn rho(n: &Integer, k: u64, proof: bool) -> LimitedFactorization {
    let one = Integer::one();
    let f = factor_pollard_rho_limited(n, &one, &one, k, proof);
    check(n, &f);
    f
}

fn limits(n: &Integer, l: FactorizationLimits) -> LimitedFactorization {
    let f = factorization_with_limits(n, &l);
    check(n, &f);
    f
}

const P30: &str = "100000000000000000000000000319";

#[test]
fn rho_limited_leaves_large_composite() {
    // gp: factor(20000000000008900000000000837) = [100000000000031, 1; 200000000000027, 1]
    // (both factors need about 10^7 rho steps).
    let n = z("20000000000008900000000000837");
    let f = rho(&n, POLLARD_RHO_DEFAULT_ITERATIONS, true);
    assert_eq!(f.composites, vec![(n, 1)]);
    assert!(f.primes.is_empty() && f.probable_primes.is_empty());
}

#[test]
fn rho_limited_splits_small_factors() {
    // gp: factor(1040300000000000000000000003318557) =
    //     [101, 1; 103, 1; 100000000000000000000000000319, 1]
    let n = z("1040300000000000000000000003318557");
    let f = rho(&n, 8191, true);
    assert_eq!(f.primes, pairs(&[("101", 1), ("103", 1), (P30, 1)]));
    assert!(f.is_complete());
}

#[test]
fn rho_limited_proof_flag() {
    // gp: isprime(nextprime(10^29)) = 1; without the proof P30 is only a probable prime.
    let n = z("1040300000000000000000000003318557");
    let f = rho(&n, 8191, false);
    assert_eq!(f.primes, pairs(&[("101", 1), ("103", 1)]));
    assert_eq!(f.probable_primes, pairs(&[(P30, 1)]));
    assert!(f.composites.is_empty());
}

#[test]
fn rho_limited_budget() {
    // gp: factor(1000000016000000063) = [1000000007, 1; 1000000009, 1]: rho from c = 1,
    // x0 = 1 needs 50430 steps.
    let n = z("1000000016000000063");
    assert_eq!(rho(&n, 8191, true).composites, vec![(n.clone(), 1)]);
    assert_eq!(
        rho(&n, 1 << 20, true).primes,
        pairs(&[("1000000007", 1), ("1000000009", 1)])
    );
}

#[test]
fn rho_limited_powers_signs_and_units() {
    // gp: factor(42525) = [3, 5; 5, 2; 7, 1]
    let f = rho(&Integer::from(-42525), 8191, true);
    assert_eq!(f.sign, -1);
    assert_eq!(f.primes, pairs(&[("3", 5), ("5", 2), ("7", 1)]));
    // gp: factor(1000000000078000000001521) = Mat([1000000000039, 2])
    let f = rho(&z("1000000000078000000001521"), 8191, true);
    assert_eq!(f.primes, pairs(&[("1000000000039", 2)]));
    // gp: factor(2^64+1) = [274177, 1; 67280421310721, 1]
    let f = rho(&(Integer::from(2).pow(64) + Integer::one()), 8191, true);
    assert_eq!(f.primes, pairs(&[("274177", 1), ("67280421310721", 1)]));
    for (n, sign) in [(0i64, 0i8), (1, 1), (-1, -1)] {
        let f = rho(&Integer::from(n), 8191, true);
        assert_eq!(f.sign, sign);
        assert!(f.primes.is_empty() && f.is_complete());
    }
}

/// A prime power costs one perfect-power test, not one split per exponent.
#[test]
fn rho_limited_prime_powers() {
    // gp: factor(3^2000) = Mat([3, 2000]); factor(2^1000*3^500) = [2, 1000; 3, 500];
    //     factor(1000003^50) = Mat([1000003, 50])
    let got = within(5, || {
        let one = Integer::one();
        [
            Integer::from(3).pow(2000),
            Integer::from(2).pow(1000) * Integer::from(3).pow(500),
            Integer::from(1000003).pow(50),
        ]
        .map(|n| factor_pollard_rho_limited(&n, &one, &one, 8191, true))
    });
    assert_eq!(got[0].primes, vec![(Integer::from(3), 2000)]);
    assert_eq!(
        got[1].primes,
        vec![(Integer::from(2), 1000), (Integer::from(3), 500)]
    );
    assert_eq!(got[2].primes, vec![(Integer::from(1000003), 50)]);
    assert!(got.iter().all(|f| f.is_complete()));
}

/// psi_12 passes the twelve strong tests 2..37 but not Baillie-PSW, so it is a composite to
/// split, not a prime. Rho from c = 1, x0 = 1 needs 450558 steps.
#[test]
fn rho_limited_psi12() {
    // gp: factor(318665857834031151167461) = [399165290221, 1; 798330580441, 1]
    let n = z("318665857834031151167461");
    assert_eq!(rho(&n, 8191, true).composites, vec![(n.clone(), 1)]);
    assert_eq!(
        rho(&n, 1 << 22, true).primes,
        pairs(&[("399165290221", 1), ("798330580441", 1)])
    );
}

/// 101 * nextprime(10^19) * nextprime(3*10^19).
const N101: &str = "30300000000000000195940000000000000211191";

#[test]
fn limits_default_is_complete() {
    // gp: factor(30300000000000000195940000000000000211191) =
    //     [101, 1; 10000000000000000051, 1; 30000000000000000041, 1]
    let f = limits(&z(N101), FactorizationLimits::default());
    assert_eq!(
        f.primes,
        pairs(&[
            ("101", 1),
            ("10000000000000000051", 1),
            ("30000000000000000041", 1)
        ])
    );
    assert!(f.is_complete());
    assert_eq!(
        FactorizationLimits::default(),
        FactorizationLimits {
            trial_division_limit: 10000,
            pollard_rho_limit: 8191,
            mpqs_limit: None,
            proof: true
        }
    );
}

#[test]
fn limits_mpqs_limit() {
    // gp: #Str(10000000000000000051 * 30000000000000000041) = 39
    let n = z(N101);
    for x in [0u32, 24, 25, 38] {
        let f = limits(
            &n,
            FactorizationLimits {
                mpqs_limit: Some(x),
                ..FactorizationLimits::default()
            },
        );
        assert_eq!(f.primes, pairs(&[("101", 1)]), "mpqs_limit {x}");
        assert_eq!(
            f.composites,
            pairs(&[("300000000000000001940000000000000002091", 1)]),
            "mpqs_limit {x}"
        );
    }
    let f = limits(
        &n,
        FactorizationLimits {
            mpqs_limit: Some(39),
            ..FactorizationLimits::default()
        },
    );
    assert!(f.is_complete());
    assert_eq!(f.primes.len(), 3);
}

#[test]
fn limits_mpqs_limit_below_25_disables_sieve() {
    // gp: #Str(1000000016000000063) = 19; factor(%) = [1000000007, 1; 1000000009, 1]
    let n = z("1000000016000000063");
    let l = |x| FactorizationLimits {
        trial_division_limit: 10000,
        pollard_rho_limit: 0,
        mpqs_limit: Some(x),
        proof: true,
    };
    assert_eq!(limits(&n, l(24)).composites, vec![(n.clone(), 1)]);
    assert_eq!(
        limits(&n, l(25)).primes,
        pairs(&[("1000000007", 1), ("1000000009", 1)])
    );
}

#[test]
fn limits_trial_division_respects_limit() {
    // gp: factor(79839848800024750353128) = [2, 3; 9973, 1; 10007, 1; 100000000000031, 1];
    //     precprime(10000) = 9973, so 10007 stays in the cofactor.
    let f = limits(
        &z("79839848800024750353128"),
        FactorizationLimits {
            trial_division_limit: 10000,
            pollard_rho_limit: 0,
            mpqs_limit: Some(0),
            proof: true,
        },
    );
    assert_eq!(f.primes, pairs(&[("2", 3), ("9973", 1)]));
    assert_eq!(f.composites, pairs(&[("1000700000000310217", 1)]));
}

/// A huge trial-division limit is bounded by isqrt of the cofactor and by the early stop on a
/// probable-prime cofactor; memory is O(1) in the limit.
#[test]
fn limits_trial_division_huge_limit() {
    let got = within(5, || {
        let mut out = Vec::new();
        for limit in [u64::MAX, 1_000_000_000_000] {
            for n in ["100", "200000000000000000000000000638", "-1", "0"] {
                let n = z(n);
                let f = factorization_with_limits(
                    &n,
                    &FactorizationLimits {
                        trial_division_limit: limit,
                        ..FactorizationLimits::default()
                    },
                );
                out.push((n, f));
            }
        }
        out
    });
    for (n, f) in &got {
        check(n, f);
    }
    for chunk in got.chunks(4) {
        // gp: factor(100) = [2, 2; 5, 2]; isprime(nextprime(10^29)) = 1
        assert_eq!(chunk[0].1.primes, pairs(&[("2", 2), ("5", 2)]));
        assert_eq!(chunk[1].1.primes, pairs(&[("2", 1), (P30, 1)]));
        assert_eq!(chunk[2].1.sign, -1);
        assert!(chunk[2].1.primes.is_empty() && chunk[2].1.is_complete());
        assert_eq!(chunk[3].1.sign, 0);
        assert!(chunk[3].1.primes.is_empty() && chunk[3].1.is_complete());
    }
}

#[test]
fn limits_proof_flag() {
    // gp: isprime(nextprime(10^29)) = 1
    let n = z("200000000000000000000000000638");
    let f = limits(
        &n,
        FactorizationLimits {
            proof: false,
            ..FactorizationLimits::default()
        },
    );
    assert_eq!(f.primes, pairs(&[("2", 1)]));
    assert_eq!(f.probable_primes, pairs(&[(P30, 1)]));
    let f = limits(&n, FactorizationLimits::default());
    assert_eq!(f.primes, pairs(&[("2", 1), (P30, 1)]));
    assert!(f.is_complete());
}

#[test]
fn limits_two_25_digit_primes() {
    // gp: factor(5000000000000000000000076000000000000000000000287) =
    //     [1000000000000000000000007, 1; 5000000000000000000000041, 1]
    let f = limits(
        &z("5000000000000000000000076000000000000000000000287"),
        FactorizationLimits::default(),
    );
    assert_eq!(
        f.primes,
        pairs(&[
            ("1000000000000000000000007", 1),
            ("5000000000000000000000041", 1)
        ])
    );
    assert!(f.is_complete());
}
