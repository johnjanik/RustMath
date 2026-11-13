//! Signs, roots and generalised binomials of `Integer` against PARI/GP and GAP reference values.
//!
//! MAGMA Handbook ch. 18 (digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`):
//! - §18.6 `Gcd`, `Lcm`, `Xgcd`: gcd and lcm are non-negative; `Xgcd` returns g >= 0 with
//!   g = x·m + y·n, and g = 0 only for (0, 0).
//! - §18.4.8 `Iroot`: the floor of the n-th root (extended to negative self and odd n).
//! - §18.4.5 `IsPower(n, k)` and `IsPower(n)` (negative inputs as PARI/GP `ispower`).
//! - §18.8 `Binomial(n, r)`, generalised to negative n by Graham-Knuth-Patashnik, *Concrete
//!   Mathematics*, eq. (5.14): C(n, k) = (-1)^k C(k-n-1, k); C(n, k) = 0 for k < 0.
//! - §18.12.1 `Modexp` on machine words (`ArithInt`/`ArithLLong::pow_mod`).
//!
//! Algorithms: Euclid and extended Euclid (D. E. Knuth, TAOCP vol. 2, 3rd ed., §4.5.2);
//! perfect-power detection by prime exponents q <= log2 n (D. J. Bernstein, Math. Comp. 67
//! (1998) 1253-1283, §1).
//!
//! References: PARI/GP 2.15.4 (`gp -q`; the command is in the comment next to each inline
//! value) and GAP 4.12.1. The fixtures in `tests/data/` come from
//! `tests/data/gen_signs_roots.gp` (`setrand(20261108)`) and, for `isqrt_big.csv`,
//! `ispower_root.csv`, `xgcd_lehmer.csv` and `binom_big.csv`, from `tests/data/gen_i3.gp`
//! (`setrand(20261112)`). Run either in an empty directory, because `write()` appends.

use rustmath_core::{EuclideanDomain, MathError};
use rustmath_integers::{ArithInt, ArithLLong, Integer, SageInteger};
use std::sync::mpsc;
use std::time::Duration;

fn z(s: &str) -> Integer {
    Integer::from_decimal_str(s.trim()).unwrap_or_else(|| panic!("bad integer {s:?}"))
}

fn i(n: i64) -> Integer {
    Integer::from(n)
}

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').collect())
}

fn binom(n: i64, k: i64) -> Integer {
    Integer::binomial(&i(n), &i(k))
}

/// 2^127 - 1, the modulus of the residue columns in the second fixture set.
fn m127() -> Integer {
    i(2).pow(127) - i(1)
}

/// Run `f` on a thread and fail if it does not finish within `secs` seconds. A panic inside
/// `f` (a failed assertion) is re-raised here with its own message, not reported as a timeout.
/// (A copy of the helper in rustmath-polynomials/tests/ch23_quotrem_gcd.rs.) The cost tests
/// below use it so that a quadratic algorithm fails instead of passing slowly; each bound is
/// at least 10 times the measured debug time.
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

// ---------------------------------------------------------------------------------------------
// T1: Binomial(n, r) for every integer n, r (§18.8)
// ---------------------------------------------------------------------------------------------

#[test]
fn binomial_negative_upper_index_inline() {
    let cases: [(i64, i64, i64); 15] = [
        (-1, 2, 1),         // gp: binomial(-1,2) -> 1
        (-1, 0, 1),         // gp: binomial(-1,0) -> 1
        (-1, 1, -1),        // gp: binomial(-1,1) -> -1
        (-1, -1, 0),        // gp: binomial(-1,-1) -> 0
        (-4, -4, 0),        // gp: binomial(-4,-4) -> 0
        (-3, 2, 6),         // gp: binomial(-3,2) -> 6
        (-2, 3, -4),        // gp: binomial(-2,3) -> -4
        (5, -1, 0),         // gp: binomial(5,-1) -> 0
        (0, -1, 0),         // gp: binomial(0,-1) -> 0
        (-150, 3, -573800), // gp: binomial(-150,3) -> -573800
        (-2, 101, -102),    // gp: binomial(-2,101) -> -102
        (-3, 150, 11476),   // gp: binomial(-3,150) -> 11476
        (-7, 30, 1947792),  // gp: binomial(-7,30) -> 1947792
        (-1, 1000, 1),      // gp: binomial(-1,1000) -> 1
        (-2, 1000, 1001),   // gp: binomial(-2,1000) -> 1001
    ];
    for (n, k, want) in cases {
        assert_eq!(binom(n, k), i(want), "binomial({n}, {k})");
    }
    // A negative upper index with a large lower index: n' = k - n - 1 = 269, k' = 120.
    // gp: binomial(-150,120)
    assert_eq!(
        binom(-150, 120),
        z("9683944761496930242351916727737098504947201826677276966180153225590958432677200")
    );
    // gp: binomial(-10^20,3)
    assert_eq!(
        Integer::binomial(&(-i(10).pow(20)), &i(3)),
        z("-166666666666666666671666666666666666666700000000000000000000")
    );
    // GAP 4.12.1: Binomial(-1,2) -> 1, Binomial(-1,-1) -> 0, Binomial(-3,-5) -> 0,
    // Binomial(-7,4) -> 210, Binomial(5,-1) -> 0.
    assert_eq!(binom(-3, -5), i(0));
    assert_eq!(binom(-7, 4), i(210));
    // Sage documentation (sage.arith.misc.binomial): "If m < 0, return 0"; binomial(3, -1) = 0,
    // binomial(-5, -2) = 0, binomial(-2, 5) = -6. gp and GAP agree:
    // [binomial(3,-1), binomial(-5,-2), binomial(-2,5)] -> [0, 0, -6].
    assert_eq!(binom(3, -1), i(0));
    assert_eq!(binom(-5, -2), i(0));
    assert_eq!(binom(-2, 5), i(-6));
}

#[test]
fn binomial_fixture() {
    // n in -30..30, k in -5..30: gp binomial(n, k).
    let mut count = 0;
    for r in rows(include_str!("data/binom_int_grid.csv")) {
        let (n, k) = (z(r[0]), z(r[1]));
        assert_eq!(Integer::binomial(&n, &k), z(r[2]), "row {r:?}");
        count += 1;
    }
    assert_eq!(count, 61 * 36);
}

#[test]
fn binomial_identities() {
    // Pascal's rule holds for the generalised binomial for all integer n, k, and the values
    // agree with the PARI-verified binomial_mod (tests/ch18_residues.rs T7) modulo small primes.
    for n in -30i64..=30 {
        for k in -5i64..=30 {
            let c = binom(n, k);
            assert_eq!(
                c,
                binom(n - 1, k - 1) + binom(n - 1, k),
                "Pascal at ({n}, {k})"
            );
            for p in [2i64, 3, 5, 7, 13] {
                assert_eq!(
                    Ok(c.modulo(&i(p))),
                    i(n).binomial_mod(&i(k), &i(p)),
                    "binomial({n}, {k}) mod {p}"
                );
            }
        }
    }
}

/// Checks the rows (n, k, binomial(n, k) % (2^127 - 1), bits of |binomial(n, k)|) of
/// `binom_big.csv` selected by `keep`; returns the number checked.
fn check_binom_big_rows(keep: impl Fn(&Integer, &Integer) -> bool) -> usize {
    let mut count = 0;
    for r in rows(include_str!("data/binom_big.csv")) {
        let (n, k) = (z(r[0]), z(r[1]));
        if !keep(&n, &k) {
            continue;
        }
        let c = Integer::binomial(&n, &k);
        assert_eq!(c.modulo(&m127()), z(r[2]), "C({n}, {k}) mod 2^127 - 1");
        assert_eq!(
            c.bit_length(),
            r[3].trim().parse::<u64>().unwrap(),
            "bits of C({n}, {k})"
        );
        count += 1;
    }
    count
}

#[test]
fn binomial_big_fixture() {
    // 80 gp rows: 40 random n < 10^6 with random k, 10 central ones, 10 with n = 10^30 + 7 + i
    // (the BigUint window), 6 with n = 2^64 - 59 + i (across the u64 boundary), 8 with n < 0,
    // (524288, 262144), (524288, 262143), (10^6, 500000), (5, -3), (-5, -3), (100, 101).
    // About 5 s in a debug build; the K-step loop needs minutes there.
    let count = within(60, || check_binom_big_rows(|_, _| true));
    assert_eq!(count, 80);
}

#[test]
fn binomial_central_rows_within_10s() {
    // C(524288, 262144), C(524288, 262143), C(10^6, 500000): 0.7 s together in a debug build.
    // The K-step loop took 10.7 s in release for the first row alone.
    let count = within(10, || {
        check_binom_big_rows(|n, k| {
            [(524_288, 262_144), (524_288, 262_143), (1_000_000, 500_000)]
                .iter()
                .any(|&(a, b)| *n == i(a) && *k == i(b))
        })
    });
    assert_eq!(count, 3);
}

#[test]
fn binomial_beyond_2_pow_32_bits_is_refused() {
    // K = min(k, N - k) and the bound K·max(1, bits(N) - 1 - bits(K)) < log2 C(N, K):
    // (2^64, 2^40): K = 2^40 > 2^32; (10^30, 10^9): bound 10^9·69; (2^70, 2^68): K = 2^68.
    // Each is refused before any allocation. The K-step loop did not return on these.
    within(10, || {
        let cases = [
            (i(2).pow(64), i(2).pow(40)),
            (i(10).pow(30), i(10).pow(9)),
            (i(2).pow(70), i(2).pow(68)),
        ];
        for (n, k) in cases {
            let res = std::panic::catch_unwind(|| Integer::binomial(&n, &k));
            let payload = res.expect_err("refused");
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|m| m.to_string()))
                .unwrap_or_default();
            assert!(msg.contains("more than 2^32 bits"), "C({n}, {k}): {msg:?}");
        }
        // Boundary guard: a small window on a huge n is computed (bound 3·31 = 93).
        // gp: binomial(2^33,3) -> 105637549982125628646836142080
        assert_eq!(
            Integer::binomial(&i(2).pow(33), &i(3)),
            z("105637549982125628646836142080")
        );
    });
}

/// The K-step loop that `SageInteger::binomial` used before the prime splitting, as an
/// independent reference: result = C(N, i)·(N - i) / (i + 1), exact at each step.
fn binomial_by_product_formula(n: i64, k: i64) -> Integer {
    if k < 0 {
        return i(0);
    }
    let (nn, negate) = if n < 0 {
        (k - n - 1, k % 2 == 1)
    } else {
        (n, false)
    };
    if k > nn {
        return i(0);
    }
    let kk = k.min(nn - k);
    let mut r = i(1);
    for j in 0..kk {
        r = r * i(nn - j) / i(j + 1);
    }
    if negate {
        -r
    } else {
        r
    }
}

#[test]
fn binomial_agrees_with_the_product_formula() {
    for n in -40i64..=300 {
        for k in -3..=n.abs() + 2 {
            assert_eq!(
                binom(n, k),
                binomial_by_product_formula(n, k),
                "C({n}, {k})"
            );
        }
    }
}

#[test]
fn binomial_1000_500() {
    // gp: binomial(1000,500) % (2^127-1) -> 62526922371930188635637843119963540756
    assert_eq!(
        binom(1000, 500).modulo(&m127()),
        z("62526922371930188635637843119963540756")
    );
}

// ---------------------------------------------------------------------------------------------
// T2: Gcd, Lcm, Xgcd signs (§18.6)
// ---------------------------------------------------------------------------------------------

#[test]
fn gcd_lcm_xgcd_inline() {
    // gp: [gcd(3,-7),gcd(4,-6),gcd(-4,6),gcd(-4,-6),gcd(0,-5),gcd(-5,0),gcd(0,0)]
    //     -> [1, 2, 2, 2, 5, 5, 0]
    let gcds: [(i64, i64, i64); 7] = [
        (3, -7, 1),
        (4, -6, 2),
        (-4, 6, 2),
        (-4, -6, 2),
        (0, -5, 5),
        (-5, 0, 5),
        (0, 0, 0),
    ];
    for (a, b, g) in gcds {
        assert_eq!(i(a).gcd(&i(b)), i(g), "gcd({a}, {b})");
    }
    // gp: [lcm(4,-6),lcm(-4,6),lcm(-4,-6),lcm(0,-5),lcm(0,0),lcm(-3,0)] -> [12, 12, 12, 0, 0, 0]
    let lcms: [(i64, i64, i64); 6] = [
        (4, -6, 12),
        (-4, 6, 12),
        (-4, -6, 12),
        (0, -5, 0),
        (0, 0, 0),
        (-3, 0, 0),
    ];
    for (a, b, l) in lcms {
        assert_eq!(i(a).lcm(&i(b)), i(l), "lcm({a}, {b})");
    }
    // gcdext(a, b) = [u, v, d] in gp; extended_gcd returns (d, u, v).
    let xgcds: [(i64, i64, (i64, i64, i64)); 7] = [
        (3, -7, (1, -2, -1)), // gp: gcdext(3,-7) -> [-2, -1, 1]
        (-4, -6, (2, 1, -1)), // gp: gcdext(-4,-6) -> [1, -1, 2]
        (0, -5, (5, 0, -1)),  // gp: gcdext(0,-5) -> [0, -1, 5]
        (-5, 0, (5, -1, 0)),  // gp: gcdext(-5,0) -> [-1, 0, 5]
        (5, -5, (5, 0, -1)),  // gp: gcdext(5,-5) -> [0, -1, 5]
        (-5, 5, (5, 0, 1)),   // gp: gcdext(-5,5) -> [0, 1, 5]
        (0, 0, (0, 0, 0)),    // gp: gcdext(0,0) -> [0, 0, 0]
    ];
    for (a, b, (d, u, v)) in xgcds {
        assert_eq!(
            i(a).extended_gcd(&i(b)),
            (i(d), i(u), i(v)),
            "extended_gcd({a}, {b})"
        );
    }
}

/// Checks one row against PARI and against the defining properties, independently of PARI's
/// choice of multipliers.
fn check_gcd_row(a: &Integer, b: &Integer, g: &Integer, l: &Integer, uv: (&Integer, &Integer)) {
    let ctx = format!("({a}, {b})");
    assert_eq!(&a.gcd(b), g, "gcd {ctx}");
    assert_eq!(&a.lcm(b), l, "lcm {ctx}");
    let (d, x, y) = a.extended_gcd(b);
    assert_eq!((&d, &x, &y), (g, uv.0, uv.1), "xgcd {ctx}");
    assert!(d.signum() >= 0, "xgcd sign {ctx}");
    assert_eq!(x * a.clone() + y * b.clone(), d, "Bezout {ctx}");
    assert_eq!(
        g.clone() * l.clone(),
        (a.clone() * b.clone()).abs(),
        "gcd·lcm {ctx}"
    );
    assert_eq!(
        d.is_zero(),
        a.is_zero() && b.is_zero(),
        "g = 0 iff (0, 0) {ctx}"
    );
}

#[test]
fn gcd_grid_fixture() {
    // a, b in -30..30: gp gcd, lcm and gcdext.
    let mut count = 0;
    for r in rows(include_str!("data/gcd_grid.csv")) {
        let v: Vec<Integer> = r.iter().map(|s| z(s)).collect();
        check_gcd_row(&v[0], &v[1], &v[2], &v[3], (&v[5], &v[6]));
        assert_eq!(v[4], v[2], "gcdext d = gcd, row {r:?}");
        count += 1;
    }
    assert_eq!(count, 61 * 61);
}

#[test]
fn xgcd_big_fixture() {
    // 300 random pairs (130 and 100 bits, a third with a common 40-bit factor, every tenth
    // with b = 0): a, b, d, u, v, lcm.
    let mut count = 0;
    for r in rows(include_str!("data/xgcd_big.csv")) {
        let v: Vec<Integer> = r.iter().map(|s| z(s)).collect();
        check_gcd_row(&v[0], &v[1], &v[2], &v[5], (&v[3], &v[4]));
        count += 1;
    }
    assert_eq!(count, 300);
}

#[test]
fn gcd_through_euclidean_domain_trait() {
    // Generic code over R: EuclideanDomain sees the same values (gp: gcd(4,-6) -> 2,
    // lcm(-4,-6) -> 12, gcdext(-4,-6) -> [1, -1, 2]).
    assert_eq!(<Integer as EuclideanDomain>::gcd(&i(4), &i(-6)), i(2));
    assert_eq!(<Integer as EuclideanDomain>::lcm(&i(-4), &i(-6)), i(12));
    assert_eq!(
        <Integer as EuclideanDomain>::extended_gcd(&i(-4), &i(-6)),
        (i(2), i(1), i(-1))
    );
}

#[test]
fn gcd_with_a_small_operand_is_one_division() {
    // gcd(m, 1) and gcd(m, p) for a 100,000-bit m: a single division with Euclid. (Binary
    // gcd would need about 10^5 shift-and-subtract passes over m.) gp: gcd(2^100000+1, 1) -> 1;
    // gcd(3^63093, 1000000007) -> 1 (1000000007 is prime and not 3).
    let m = i(2).pow(100_000) + i(1);
    assert_eq!(m.gcd(&i(1)), i(1));
    assert_eq!(i(1).gcd(&m), i(1));
    assert_eq!((-m.clone()).lcm(&i(1)), m);
    let m3 = i(3).pow(63_093);
    assert_eq!(m3.gcd(&i(1_000_000_007)), i(1));
    assert_eq!(m3.gcd(&i(-81)), i(81)); // gp: gcd(3^63093, -81) -> 81
}

#[test]
fn xgcd_lehmer_fixture() {
    // 24 gp rows a, b, d, u, v (gcdext(a, b) = [u, v, d]) of 65 to 3000 bits: Fibonacci pairs
    // with b negated, b = -a, b = 7a, an unbalanced 70-bit b, a = 0, b = 3·2^(bits+5), common
    // 200-bit factors. Lehmer's matrix steps must give Euclid's (and PARI's) multipliers.
    let mut count = 0;
    for r in rows(include_str!("data/xgcd_lehmer.csv")) {
        let v: Vec<Integer> = r.iter().map(|s| z(s)).collect();
        let (a, b, d) = (&v[0], &v[1], &v[2]);
        let l = if a.is_zero() || b.is_zero() {
            Integer::zero()
        } else {
            (a.clone() * b.clone()).abs() / d.clone()
        };
        check_gcd_row(a, b, d, &l, (&v[3], &v[4]));
        count += 1;
    }
    assert_eq!(count, 24);
}

/// A deterministic pseudo-random Integer of exactly `bits` bits (an LCG; `bits >= 1`).
fn lcg_integer(bits: u64, st: &mut u64) -> Integer {
    use num_bigint::BigUint;
    let words = bits.div_ceil(64);
    let mut v: Vec<u32> = Vec::new();
    for _ in 0..words {
        *st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push(*st as u32);
        v.push((*st >> 32) as u32);
    }
    let mut n = BigUint::new(v) >> (words * 64 - bits);
    n.set_bit(bits - 1, true);
    Integer::new(n.into())
}

fn stein_gcd(a: &Integer, b: &Integer) -> Integer {
    // num-integer's BigUint gcd is Stein's binary algorithm: an implementation independent of
    // the crate's Lehmer code.
    let g = num_integer::Integer::gcd(a.as_bigint().magnitude(), b.as_bigint().magnitude());
    Integer::new(g.into())
}

#[test]
fn lehmer_gcd_agrees_with_stein_and_extended_euclid() {
    // 400 LCG pairs up to 3000 bits: a third with a common factor, some b = a, b = k·a, a = 0,
    // random signs. gcd against Stein's binary gcd; extended_gcd against num-integer's
    // truncated-quotient extended Euclid on BigInt (the algorithm before Lehmer, whose
    // triples equal gp gcdext on the fixtures above).
    let mut st = 4242u64;
    let next = |m: u64, st: &mut u64| {
        *st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*st >> 17) % m
    };
    for idx in 0..400 {
        let ba = 1 + next(3000, &mut st);
        let bb = 1 + next(3000, &mut st);
        let mut a = lcg_integer(ba, &mut st);
        let mut b = lcg_integer(bb, &mut st);
        if idx % 3 == 0 {
            let bc = 1 + next(1000, &mut st);
            let c = lcg_integer(bc, &mut st);
            a = a * c.clone();
            b = b * c;
        }
        if idx % 7 == 0 {
            b = a.clone();
        }
        if idx % 11 == 0 {
            b = a.clone() * i(1 + next(1000, &mut st) as i64);
        }
        if idx % 13 == 0 {
            a = Integer::zero();
        }
        if next(2, &mut st) == 1 {
            a = -a;
        }
        if next(2, &mut st) == 1 {
            b = -b;
        }
        assert_eq!(a.gcd(&b), stein_gcd(&a, &b), "gcd case {idx}");
        let want = if a.is_zero() && b.is_zero() {
            (Integer::zero(), Integer::zero(), Integer::zero())
        } else {
            let e = num_integer::Integer::extended_gcd(a.as_bigint(), b.as_bigint());
            (Integer::new(e.gcd), Integer::new(e.x), Integer::new(e.y))
        };
        assert_eq!(a.extended_gcd(&b), want, "extended_gcd case {idx}");
    }
}

#[test]
fn xgcd_of_large_operands() {
    // a, b of 30,000 bits with a common 5,000-bit factor: Bezout, g | a, g | b and
    // gcd(a/g, b/g) = 1 by Stein.
    let mut st = 99u64;
    let c = lcg_integer(5_000, &mut st);
    let a = lcg_integer(25_000, &mut st) * c.clone();
    let b = -(lcg_integer(24_990, &mut st) * c.clone());
    let (g, x, y) = a.extended_gcd(&b);
    assert_eq!(x * a.clone() + y * b.clone(), g);
    assert!((a.clone() % g.clone()).is_zero() && (b.clone() % g.clone()).is_zero());
    assert!(stein_gcd(&(a.clone() / g.clone()), &(b.clone() / g.clone())).is_one());
    assert!((g.clone() % c).is_zero());
    assert_eq!(g, a.gcd(&b));
}

#[test]
fn gcd_and_xgcd_of_100k_bit_operands() {
    // Speed test. Debug build: Lehmer gcd 0.14 s and xgcd 0.3 s; Euclid took 9.7 s and the
    // num-integer extended Euclid 11.3 s. Only the two Lehmer calls are under the bound; the
    // Stein comparison (0.9 s in debug) and the identities run after it.
    let mut st = 2026u64;
    let a = lcg_integer(100_000, &mut st);
    let b = -lcg_integer(99_993, &mut st);
    let (a2, b2) = (a.clone(), b.clone());
    let (g, (d, x, y)) = within(5, move || {
        let g = <Integer as EuclideanDomain>::gcd(&a2, &b2);
        let e = <Integer as EuclideanDomain>::extended_gcd(&a2, &b2);
        (g, e)
    });
    assert_eq!(g, d);
    assert!((a.clone() % g.clone()).is_zero() && (b.clone() % g.clone()).is_zero());
    assert_eq!(x * a.clone() + y * b.clone(), g);
    assert_eq!(g, stein_gcd(&a, &b));
}

#[test]
fn gcd_grid_fixture_through_i32_and_i64() {
    // The machine-word EuclideanDomain impls of rustmath-core (MAGMA Gcd, Lcm, Xgcd, §18.6)
    // on the gp grid a, b in -30..30 (columns a, b, gcd, lcm, d, u, v; gcdext = [u, v, d]).
    let mut count = 0;
    for r in rows(include_str!("data/gcd_grid.csv")) {
        let v: Vec<i64> = r.iter().map(|s| s.trim().parse().unwrap()).collect();
        let (a, b) = (v[0], v[1]);
        assert_eq!(<i64 as EuclideanDomain>::gcd(&a, &b), v[2], "i64 gcd {r:?}");
        assert_eq!(<i64 as EuclideanDomain>::lcm(&a, &b), v[3], "i64 lcm {r:?}");
        assert_eq!(
            <i64 as EuclideanDomain>::extended_gcd(&a, &b),
            (v[4], v[5], v[6]),
            "i64 xgcd {r:?}"
        );
        let n = |x: i64| i32::try_from(x).unwrap();
        let (a32, b32) = (n(a), n(b));
        assert_eq!(
            <i32 as EuclideanDomain>::gcd(&a32, &b32),
            n(v[2]),
            "i32 gcd {r:?}"
        );
        assert_eq!(
            <i32 as EuclideanDomain>::lcm(&a32, &b32),
            n(v[3]),
            "i32 lcm {r:?}"
        );
        assert_eq!(
            <i32 as EuclideanDomain>::extended_gcd(&a32, &b32),
            (n(v[4]), n(v[5]), n(v[6])),
            "i32 xgcd {r:?}"
        );
        count += 1;
    }
    assert_eq!(count, 61 * 61);
}

#[test]
fn machine_word_gcd_agrees_with_integer() {
    // 100,000 LCG pairs (values shifted right by a random amount; every 17th b = i64::MIN,
    // every 19th b = 0) against Integer gcd / lcm / extended_gcd, whenever the Integer result
    // fits in i64. Integer is fixture-tested against gp above, and its extended_gcd is checked
    // here against num-integer's truncated extended Euclid on BigInt as well.
    let fits = |x: &Integer| i64::try_from(x).ok();
    let mut st = 5u64;
    let mut next = || {
        st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        st
    };
    let mut checked = 0;
    for idx in 0..100_000 {
        let r = next();
        let a = (r as i64) >> (r % 50);
        let r = next();
        let mut b = (r as i64) >> (r % 50);
        if idx % 17 == 0 {
            b = i64::MIN;
        }
        if idx % 19 == 0 {
            b = 0;
        }
        let (ia, ib) = (i(a), i(b));
        let g = ia.gcd(&ib);
        let Some(g64) = fits(&g) else { continue };
        assert_eq!(<i64 as EuclideanDomain>::gcd(&a, &b), g64, "gcd({a}, {b})");
        let (d, x, y) = ia.extended_gcd(&ib);
        if !(a == 0 && b == 0) {
            let e = num_integer::Integer::extended_gcd(ia.as_bigint(), ib.as_bigint());
            let ne = (Integer::new(e.gcd), Integer::new(e.x), Integer::new(e.y));
            assert_eq!(
                (d.clone(), x.clone(), y.clone()),
                ne,
                "num-integer xgcd({a}, {b})"
            );
        }
        let want = (fits(&d).unwrap(), fits(&x).unwrap(), fits(&y).unwrap());
        assert_eq!(
            <i64 as EuclideanDomain>::extended_gcd(&a, &b),
            want,
            "xgcd({a}, {b})"
        );
        if let Some(l) = fits(&ia.lcm(&ib)) {
            assert_eq!(<i64 as EuclideanDomain>::lcm(&a, &b), l, "lcm({a}, {b})");
        }
        if let (Ok(a32), Ok(b32), Ok(g32)) =
            (i32::try_from(a), i32::try_from(b), i32::try_from(g64))
        {
            assert_eq!(
                <i32 as EuclideanDomain>::gcd(&a32, &b32),
                g32,
                "i32 gcd({a}, {b})"
            );
            let w32 = (
                g32,
                i32::try_from(want.1).unwrap(),
                i32::try_from(want.2).unwrap(),
            );
            assert_eq!(<i32 as EuclideanDomain>::extended_gcd(&a32, &b32), w32);
        }
        checked += 1;
    }
    assert!(checked > 99_000, "{checked}");
}

// ---------------------------------------------------------------------------------------------
// T3a: Isqrt, the floor of the square root (§18.4.8)
// ---------------------------------------------------------------------------------------------

#[test]
fn isqrt_big_fixture() {
    // 90 gp rows a, sqrtint(a) of 65 to 3000 bits; every third row is b^2, b^2 + 1 or b^2 - 1.
    let mut count = 0;
    for r in rows(include_str!("data/isqrt_big.csv")) {
        let (a, want) = (z(r[0]), z(r[1]));
        let s = a.sqrt().unwrap();
        assert_eq!(s, want, "sqrt({a})");
        assert!(s.clone() * s.clone() <= a, "floor property {a}");
        assert!((s.clone() + i(1)) * (s + i(1)) > a, "floor property {a}");
        count += 1;
    }
    assert_eq!(count, 90);
}

#[test]
fn isqrt_of_a_100k_bit_number() {
    // gp: a=sqrtint(3^63093+12345); [a % (2^127-1), #binary(a)]
    //     -> [130117212609866437730867252961009611028, 50001]
    // About 50 ms in a debug build. Newton started at x = self took 193 s in debug (13 s in
    // release) for this root.
    within(10, || {
        let a = i(3).pow(63_093) + i(12_345);
        let s = a.sqrt().unwrap();
        assert_eq!(s.bit_length(), 50_001);
        assert_eq!(
            s.modulo(&m127()),
            z("130117212609866437730867252961009611028")
        );
    });
}

// ---------------------------------------------------------------------------------------------
// T3: nth_root, the floor of the real root (§18.4.8)
// ---------------------------------------------------------------------------------------------

#[test]
fn nth_root_negative_inline() {
    // gp: fl(a,n)=if(a>=0, sqrtnint(a,n), my(t=sqrtnint(-a,n)); if(t^n==-a, -t, -t-1))
    let cases: [(i64, u32, i64); 7] = [
        (-8, 3, -2),  // gp: fl(-8,3) -> -2
        (-9, 3, -3),  // gp: fl(-9,3) -> -3 (floor; truncation would give -2)
        (-27, 3, -3), // gp: fl(-27,3) -> -3
        (-1, 3, -1),  // gp: fl(-1,3) -> -1
        (-32, 5, -2), // gp: fl(-32,5) -> -2
        (-33, 5, -3), // gp: fl(-33,5) -> -3
        (-1, 1, -1),  // gp: fl(-1,1) -> -1
    ];
    for (a, n, want) in cases {
        assert_eq!(i(a).nth_root(n), Ok(i(want)), "nth_root({a}, {n})");
    }
    let big = i(10).pow(30) + i(1);
    // gp: fl(-(10^30+1),3) -> -10000000001; fl(10^30+1,3) -> 10000000000
    assert_eq!((-big.clone()).nth_root(3), Ok(i(-10_000_000_001)));
    assert_eq!(big.nth_root(3), Ok(i(10_000_000_000)));
    // A huge odd degree returns at once: gp: fl(-5,10^9+1) -> -2
    assert_eq!(i(-5).nth_root(1_000_000_001), Ok(i(-2)));
    // Refusals (gp: sqrtnint(-4,2) -> "domain error"): even root of a negative, degree 0.
    assert!(matches!(
        i(-4).nth_root(2),
        Err(MathError::InvalidArgument(_))
    ));
    assert!(matches!(
        i(-16).nth_root(4),
        Err(MathError::InvalidArgument(_))
    ));
    assert!(matches!(
        i(0).nth_root(0),
        Err(MathError::InvalidArgument(_))
    ));
}

fn check_root_rows(csv: &str) -> usize {
    let mut count = 0;
    for r in rows(csv) {
        let a = z(r[0]);
        let n: u32 = r[1].parse().unwrap();
        let got = a.nth_root(n);
        if r[2] == "E" {
            assert!(got.is_err(), "nth_root({a}, {n}) must be Err, got {got:?}");
        } else {
            let root = got.unwrap_or_else(|e| panic!("nth_root({a}, {n}): {e:?}"));
            assert_eq!(root, z(r[2]), "nth_root({a}, {n})");
            // Floor property: r^n <= a < (r+1)^n.
            assert!(root.pow(n) <= a, "r^n <= a for ({a}, {n})");
            assert!(
                (root.clone() + i(1)).pow(n) > a,
                "a < (r+1)^n for ({a}, {n})"
            );
        }
        count += 1;
    }
    count
}

#[test]
fn nth_root_grid_fixture() {
    // a in -400..400, n in 1..9 ("E": even n and a < 0).
    assert_eq!(
        check_root_rows(include_str!("data/iroot_grid.csv")),
        801 * 9
    );
}

#[test]
fn nth_root_big_fixture() {
    // 200 random 256-bit a of both signs, n in 2..17, every fourth an exact power.
    assert_eq!(check_root_rows(include_str!("data/iroot_big.csv")), 200);
}

// ---------------------------------------------------------------------------------------------
// T4: is_power(k) (MAGMA IsPower(n, k), §18.4.5)
// ---------------------------------------------------------------------------------------------

#[test]
fn is_power_inline() {
    // gp: [ispower(-8,3),ispower(-27,3),ispower(-1,3),ispower(-1,2),ispower(-4,2),
    //      ispower(-16,4),ispower(-64,6),ispower(-64,3),ispower(0,3)] -> [1,1,1,0,0,0,0,1,1]
    let cases: [(i64, u32, bool); 9] = [
        (-8, 3, true),
        (-27, 3, true),
        (-1, 3, true),
        (-1, 2, false),
        (-4, 2, false),
        (-16, 4, false),
        (-64, 6, false),
        (-64, 3, true),
        (0, 3, true),
    ];
    for (a, k, want) in cases {
        assert_eq!(i(a).is_power(k), want, "is_power({a}, {k})");
    }
    // k = 0: b^0 = 1 for every b, so only 1 is a 0-th power (PARI refuses k <= 0).
    assert!(i(1).is_power(0));
    assert!(!i(2).is_power(0));
    // Large odd exponents stay cheap: the check works on |a|. gp: ispower(-5,10^9+1) -> 0,
    // ispower(-2^99,99) -> 1
    assert!(!i(-5).is_power(1_000_000_001));
    assert!((-i(2).pow(99)).is_power(99));
}

#[test]
fn is_power_grid_fixture() {
    // a in -400..400, k in 1..8: gp ispower(a, k).
    let mut count = 0;
    for r in rows(include_str!("data/ispower_k_grid.csv")) {
        let a = z(r[0]);
        let k: u32 = r[1].parse().unwrap();
        assert_eq!(a.is_power(k), r[2] == "1", "is_power({a}, {k})");
        count += 1;
    }
    assert_eq!(count, 801 * 8);
}

// ---------------------------------------------------------------------------------------------
// T5: is_perfect_power (MAGMA IsPower(n), §18.4.5; PARI ispower(n, , &r))
// ---------------------------------------------------------------------------------------------

#[test]
fn is_perfect_power_inline() {
    // gp: pp(a)=my(r,k=ispower(a,,&r)); [k, if(k,r,0)]
    let p = |a: Integer| a.is_perfect_power();
    assert_eq!(p(i(64)), Some((i(2), 6))); // gp: pp(64) -> [6, 2] (minimal base)
    assert_eq!(p(i(-64)), Some((i(-4), 3))); // gp: pp(-64) -> [3, -4]
    assert_eq!(p(i(-4)), None); // gp: pp(-4) -> [0, 0]
    assert_eq!(p(i(-32)), Some((i(-2), 5))); // gp: pp(-32) -> [5, -2]

    // Exponents beyond the old cap of 64 (which reported "not a power").
    assert_eq!(p(i(2).pow(67)), Some((i(2), 67))); // gp: pp(2^67) -> [67, 2]
    assert_eq!(p(-i(2).pow(67)), Some((i(-2), 67))); // gp: pp(-2^67) -> [67, -2]
    assert_eq!(p(i(2).pow(128)), Some((i(2), 128))); // gp: pp(2^128) -> [128, 2]
    assert_eq!(p(i(10).pow(60)), Some((i(10), 60))); // gp: pp(10^60) -> [60, 10]
    assert_eq!(p(-i(10).pow(60)), Some((i(-10000), 15))); // gp: pp(-10^60) -> [15, -10000]
    assert_eq!(p(-i(3).pow(41)), Some((i(-3), 41))); // gp: pp(-3^41) -> [41, -3]

    // gp: pp(0), pp(1), pp(-1), pp(12) -> [0, 0] (outside MAGMA's domain n > 1 for |n| <= 1)
    for a in [0, 1, -1, 12] {
        assert_eq!(p(i(a)), None, "is_perfect_power({a})");
    }
}

#[test]
fn is_perfect_power_sieve_inline() {
    // The power-residue sieve (q < 32) and the 2-adic root refute exponents without a floor
    // root; every answer below still matches gp. gp: pp(a)=my(r,k=ispower(a,,&r)); [k, if(k,r,0)]
    let p = |a: Integer| a.is_perfect_power();
    let two = i(2);
    // A large non-power: gp pp(2^20000+1) -> [0, 0]. Before the sieve this paid one
    // BigUint::nth_root per prime q < 20000 (about 50 s in a debug build).
    assert_eq!(p(two.pow(20000) + i(1)), None);
    // gp: pp(2^4096+1), pp(10^1000+1), pp(10^1000-1) -> [0, 0]
    assert_eq!(p(two.pow(4096) + i(1)), None);
    assert_eq!(p(i(10).pow(1000) + i(1)), None);
    assert_eq!(p(i(10).pow(1000) - i(1)), None);
    // Genuine powers pass the sieve and are found by the exact root.
    assert_eq!(p(i(3).pow(20011)), Some((i(3), 20011))); // gp: pp(3^20011) -> [20011, 3]
    assert_eq!(p(-i(3).pow(100055)), Some((i(-3), 100055))); // gp: pp(-3^100055) -> [100055, -3]
    let m61 = two.pow(61) - i(1);
    // gp: pp((2^61-1)^331) -> [331, 2^61-1]; pp(-(2^61-1)^662) -> [331, -(2^61-1)^2]
    assert_eq!(p(m61.pow(331)), Some((m61.clone(), 331)));
    assert_eq!(p(-m61.pow(662)), Some((-m61.pow(2), 331)));
    // 53599 = 7*13*19*31 is divisible by every sieve prime for q = 3 (7, 13, 19, 31), so the
    // sieve cannot refute a cube there and the exact (2-adic) root decides.
    // gp: pp(53599), pp(53599^3), pp(2*53599^3), pp(-53599^5)
    //     -> [0, 0], [3, 53599], [0, 0], [5, -53599]
    let c = i(7 * 13 * 19 * 31);
    assert_eq!(p(c.clone()), None);
    assert_eq!(p(c.pow(3)), Some((c.clone(), 3)));
    assert_eq!(p(i(2) * c.pow(3)), None);
    assert_eq!(p(-c.pow(5)), Some((-c.clone(), 5)));
}

fn check_perfect_power_rows(csv: &str) -> usize {
    let mut count = 0;
    for r in rows(csv) {
        let a = z(r[0]);
        let k: u32 = r[1].parse().unwrap();
        let want = if k == 0 { None } else { Some((z(r[2]), k)) };
        assert_eq!(a.is_perfect_power(), want, "is_perfect_power({a})");
        count += 1;
    }
    count
}

#[test]
fn is_perfect_power_grid_fixture() {
    // a in -3000..3000: gp ispower(a, , &r).
    assert_eq!(
        check_perfect_power_rows(include_str!("data/perfect_power_grid.csv")),
        6001
    );
}

#[test]
fn is_perfect_power_big_fixture() {
    // ±b^k with b < 2^20 and k in 2..71, every fifth perturbed to a non-power.
    assert_eq!(
        check_perfect_power_rows(include_str!("data/perfect_power_big.csv")),
        120
    );
}

#[test]
fn is_perfect_power_huge_inputs() {
    // gp: [ispower(2^1000000+1), ispower(3^30030), ispower(3^30030+2), ispower(2^100000)]
    //     -> [0, 30030, 0, 100000] (the root is 3, resp. 2)
    // About 0.2 s in a debug build for all four. The floor-root loop took about 12 s in debug
    // (7.5 s in release) for the first input alone.
    within(5, || {
        let p = i(2).pow(1_000_000) + i(1);
        assert_eq!(p.is_perfect_power(), None);
        let t = i(3).pow(30_030);
        assert_eq!(t.is_perfect_power(), Some((i(3), 30_030)));
        assert_eq!((t + i(2)).is_perfect_power(), None);
        assert_eq!(i(2).pow(100_000).is_perfect_power(), Some((i(2), 100_000)));
    });
}

// ---------------------------------------------------------------------------------------------
// T5b: exact_root, the root of MAGMA IsPower(n, k) (§18.4.5; PARI ispower(n, k, &r))
// ---------------------------------------------------------------------------------------------

#[test]
fn exact_root_inline() {
    // gp: r=0; [ispower(16,4,&r), r] -> [1, 2]; [ispower(-8,3,&r), r] -> [1, -2];
    //     [ispower(64,6,&r), r] -> [1, 2]; ispower(-64,6) -> 0
    assert_eq!(i(16).exact_root(4), Some(i(2)));
    assert_eq!(i(-8).exact_root(3), Some(i(-2)));
    assert_eq!(i(64).exact_root(6), Some(i(2)));
    assert_eq!(i(-64).exact_root(6), None);
    assert_eq!(i(0).exact_root(3), Some(i(0)));
    assert_eq!(i(1).exact_root(0), Some(i(1)));
    assert_eq!(i(2).exact_root(0), None);
    assert_eq!(i(-1).exact_root(2), None);
    assert_eq!(i(-1).exact_root(3), Some(i(-1)));
    assert_eq!(i(-7).exact_root(1), Some(i(-7)));
    // gp: [ispower(3^20011,20011,&r), r] -> [1, 3]; [ispower(-3^100055,100055,&r), r]
    //     -> [1, -3]; [ispower((2^61-1)^331,331,&r), r] -> [1, 2305843009213693951];
    //     [ispower(2^(64*7)*5^7,7,&r), r == 2^64*5] -> [1, 1]
    assert_eq!(i(3).pow(20_011).exact_root(20_011), Some(i(3)));
    assert_eq!((-i(3).pow(100_055)).exact_root(100_055), Some(i(-3)));
    let m61 = i(2).pow(61) - i(1);
    assert_eq!(m61.pow(331).exact_root(331), Some(m61));
    let b = i(2).pow(64) * i(5);
    assert_eq!((i(2).pow(64 * 7) * i(5).pow(7)).exact_root(7), Some(b));
    // gp: n=3^100003+2; [ispower(n,3), ispower(n,101), ispower(n,9999), ispower(n,100003)]
    //     -> [0, 0, 0, 0]. 41 ms in a debug build for all four; the floor-root test took 11 s
    //     for k = 9999 alone.
    let n = i(3).pow(100_003) + i(2);
    within(10, move || {
        for k in [3u32, 101, 9999, 100_003] {
            assert_eq!(n.exact_root(k), None, "k = {k}");
            assert!(!n.is_power(k), "k = {k}");
        }
    });
}

#[test]
fn exact_root_fixture() {
    // 2,486 gp rows a, k, ispower(a, k), r: a in -200..200 with k in 1..6, plus ±b^k
    // (b < 2^40, k in {2,3,4,5,6,7,9,12,15,30,31,64}), b^k + 2 and 2^(k·j)·b^k.
    let mut count = 0;
    for r in rows(include_str!("data/ispower_root.csv")) {
        let (a, k) = (z(r[0]), r[1].trim().parse::<u32>().unwrap());
        let want = match r[2].trim() {
            "1" => Some(z(r[3])),
            "0" => None,
            other => panic!("bad flag {other:?}"),
        };
        assert_eq!(a.exact_root(k), want, "exact_root({a}, {k})");
        assert_eq!(a.is_power(k), want.is_some(), "is_power({a}, {k})");
        if let Some(b) = want {
            assert_eq!(b.pow(k), a);
        }
        count += 1;
    }
    assert_eq!(count, 2486);
}

// ---------------------------------------------------------------------------------------------
// T5c: DivisorSigma(k, n) exponent range (§18.7)
// ---------------------------------------------------------------------------------------------

#[test]
fn sigma_exponent_overflow_is_an_error() {
    // (a + 1)·k > u32::MAX: the value would have more than 2^31 bits. Every exponent is checked
    // before any power is formed, so each call returns at once. Before, the u32 product
    // overflowed (a panic in debug, a wrapped exponent and Ok(0) in release), and 3^(2^31) was
    // formed first and did not return.
    within(10, || {
        let cases = [
            (2, 1u32 << 31),
            (8, 1 << 30),
            (1024, 400_000_000),
            (3, 1 << 31),
        ];
        for (n, k) in cases {
            assert!(
                matches!(i(n).sigma(k), Err(MathError::NumericalError(_))),
                "sigma({n}, {k})"
            );
        }
    });
}

// ---------------------------------------------------------------------------------------------
// T6: machine-word pow_mod (MAGMA Modexp, §18.12.1), same conventions as Integer::mod_pow
// ---------------------------------------------------------------------------------------------

#[test]
fn fast_arith_pow_mod_negative_exponents_inline() {
    let (a32, a64) = (ArithInt::new(), ArithLLong::new());
    // gp: pm(a,b,m)=iferr(lift(Mod(a,abs(m))^b), E, "ERR")
    let ok: [(i32, i32, i32, i32); 9] = [
        (3, -1, 7, 5),  // gp: pm(3,-1,7) -> 5
        (3, -5, 7, 3),  // gp: pm(3,-5,7) -> 3
        (2, -1, 9, 5),  // gp: pm(2,-1,9) -> 5
        (-2, -3, 7, 6), // gp: pm(-2,-3,7) -> 6
        (5, -3, 1, 0),  // gp: pm(5,-3,1) -> 0
        (0, 0, 7, 1),   // gp: pm(0,0,7) -> 1
        (0, -1, 1, 0),  // gp: pm(0,-1,1) -> 0
        (3, -2, -1, 0), // gp: pm(3,-2,-1) -> 0
        (3, -1, -7, 5), // gp: pm(3,-1,-7) -> 5
    ];
    for (a, b, m, want) in ok {
        assert_eq!(
            a32.pow_mod(a, b, m),
            Ok(want),
            "ArithInt::pow_mod({a}, {b}, {m})"
        );
        assert_eq!(
            a64.pow_mod(a.into(), b.into(), m.into()),
            Ok(i64::from(want)),
            "ArithLLong::pow_mod({a}, {b}, {m})"
        );
    }
    // gp: Mod(6,9)^-1, Mod(0,7)^-1 -> "impossible inverse in Fp_inv"
    for (a, b, m) in [(6, -1, 9), (0, -1, 7)] {
        assert_eq!(a32.pow_mod(a, b, m), Err(MathError::NotInvertible));
        assert_eq!(
            a64.pow_mod(a.into(), b.into(), m.into()),
            Err(MathError::NotInvertible)
        );
    }
    assert_eq!(a32.pow_mod(2, 3, 0), Err(MathError::DivisionByZero));
    assert_eq!(a64.pow_mod(2, 3, 0), Err(MathError::DivisionByZero));
    assert_eq!(a64.pow_mod(2, -3, 0), Err(MathError::DivisionByZero));
}

#[test]
fn fast_arith_pow_mod_extremes() {
    let (a32, a64) = (ArithInt::new(), ArithLLong::new());
    // gp: pm(3,-1,-2^63) -> 3074457345618258603
    assert_eq!(a64.pow_mod(3, -1, i64::MIN), Ok(3_074_457_345_618_258_603));
    // gp: pm(3,-2^63,2^63-1) -> 6289343986027528696
    assert_eq!(
        a64.pow_mod(3, i64::MIN, i64::MAX),
        Ok(6_289_343_986_027_528_696)
    );
    // gp: pm(-5,-2^63+1,-2^63) -> 9223372036854775803
    assert_eq!(
        a64.pow_mod(-5, i64::MIN + 1, i64::MIN),
        Ok(9_223_372_036_854_775_803)
    );
    // gp: pm(-2^63,-2^63,2^63-1) -> 1
    assert_eq!(a64.pow_mod(i64::MIN, i64::MIN, i64::MAX), Ok(1));
    // gp: pm(2,-1,-2^63) -> "impossible inverse"
    assert_eq!(a64.pow_mod(2, -1, i64::MIN), Err(MathError::NotInvertible));
    // gp: pm(123456,-789,1000000007) -> 548745582
    assert_eq!(a64.pow_mod(123_456, -789, 1_000_000_007), Ok(548_745_582));
    // gp: pm(10^18,-10^18,2^63-25) -> 3517231073822784680
    assert_eq!(
        a64.pow_mod(
            1_000_000_000_000_000_000,
            -1_000_000_000_000_000_000,
            i64::MAX - 24
        ),
        Ok(3_517_231_073_822_784_680)
    );
    // gp: pm(3,-2^31,-2^31) -> 1
    assert_eq!(a32.pow_mod(3, i32::MIN, i32::MIN), Ok(1));
    // gp: pm(-7,-12345,2^31-1) -> 1829713835
    assert_eq!(a32.pow_mod(-7, -12_345, i32::MAX), Ok(1_829_713_835));
    // gp: pm(2,30,-2^31) -> 1073741824
    assert_eq!(a32.pow_mod(2, 30, i32::MIN), Ok(1_073_741_824));
    // gp: pm(-2^31,-2^31,2^31-1) -> 1
    assert_eq!(a32.pow_mod(i32::MIN, i32::MIN, i32::MAX), Ok(1));
}

#[test]
fn fast_arith_pow_mod_agrees_with_integer_mod_pow() {
    // Integer::mod_pow is fixture-tested against PARI (ch18_residues.rs T1); a PARI table of
    // lift(Mod(a,|m|)^b) over this grid (25,830 rows) matched the same rules exactly.
    let (a32, a64) = (ArithInt::new(), ArithLLong::new());
    let mut count = 0;
    for a in -20i32..=20 {
        for b in -10i32..=10 {
            for m in (-15i32..=15).filter(|&m| m != 0) {
                let want = i(a.into())
                    .mod_pow(&i(b.into()), &i(m.into()))
                    .map(|r| i32::try_from(&r).unwrap());
                assert_eq!(a32.pow_mod(a, b, m), want, "ArithInt ({a}, {b}, {m})");
                assert_eq!(
                    a64.pow_mod(a.into(), b.into(), m.into()),
                    want.map(i64::from),
                    "ArithLLong ({a}, {b}, {m})"
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 41 * 21 * 30);
}

#[test]
fn fast_arith_llong_is_prime_mersenne_61() {
    // ArithLLong::is_prime is the deterministic is_prime_u64 (fast_arith.rs).
    // gp: isprime(2^61-1) -> 1; isprime(2^61+1) -> 0
    let a64 = ArithLLong::new();
    assert!(a64.is_prime((1i64 << 61) - 1));
    assert!(!a64.is_prime((1i64 << 61) + 1));
}
