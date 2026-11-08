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
//! `tests/data/gen_signs_roots.gp` (`setrand(20261108)`; run it in an empty directory, because
//! `write()` appends).

use rustmath_core::{EuclideanDomain, MathError};
use rustmath_integers::{ArithInt, ArithLLong, Integer, SageInteger};

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
    // The power-residue sieve skips the exact root for exponents it refutes; every answer
    // below still matches gp. gp: pp(a)=my(r,k=ispower(a,,&r)); [k, if(k,r,0)]
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
    // sieve cannot refute a cube there and the exact root decides.
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
                    .map(|r| i32::try_from(r.to_i64()).unwrap());
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
fn fast_arith_miller_rabin_after_pow_mod_change() {
    // ArithLLong::is_prime runs Miller-Rabin through the internal pow_mod helper.
    // gp: isprime(2^61-1) -> 1; isprime(2^61+1) -> 0
    let a64 = ArithLLong::new();
    assert!(a64.is_prime((1i64 << 61) - 1));
    assert!(!a64.is_prime((1i64 << 61) + 1));
}
