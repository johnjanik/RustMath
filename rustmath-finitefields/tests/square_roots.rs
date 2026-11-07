//! Square roots in Z/p^kZ and Z/mZ: `square_root_mod_prime_power`,
//! `all_square_roots_mod_prime_power` and `IntegerMod::{sqrt, is_square, all_sqrts, num_sqrts}`.
//!
//! MAGMA Handbook: ch. 18 §18.12.1 `Modsqrt`; ch. 19 §19.5.6 `IsSquare`, `Sqrt` / `SquareRoot`,
//! `AllSquareRoots` / `AllSqrts` (worked example H19E4, ported as `handbook_h19e4`); ch. 47
//! §47.10 `SquareRoot` (on Z/p^kZ). Review ids `ch018.Modsqrt`, `ch047.SquareRoot`: the old
//! Hensel lift returned `None` whenever 2x was not invertible mod p (p = 2, or p | a), and
//! returned non-roots for odd p and k >= 3 (the correction was scaled by p, not p^i).
//!
//! Documented choice of root: the least non-negative root for a prime-power modulus; for a
//! composite modulus, the CRT combination of the per-prime-power least roots. PARI's
//! `issquare(x, &r)` returns some other root (17 mod 32 gives 9, the least is 7), so no test
//! compares against gp's chosen root.
//!
//! # Reference values: PARI/GP 2.15.4
//!
//! Every expected value below can be reproduced with these gp definitions (`gp -q`):
//!
//! ```text
//! \\ (B) brute force: every x in [0, m) with x^2 = a (mod m), ascending
//! roots(a,m) = select(x -> Mod(x,m)^2 == Mod(a,m), [0..m-1]);
//!
//! \\ (F) fingerprint of p^k: [#squares a in [0,p^k), sum of their least roots, total #roots]
//! \\     (from (B); also checks (B) against issquare, the total must be p^k)
//! tab(p,k) = { my(m=p^k, ns=0, sl=0, nr=0, R); for(a=0, m-1, R=roots(a,m); nr+=#R;
//!   if(#R != 0, ns++; sl+=R[1]); if((#R != 0) != issquare(Mod(a,m)), error("issquare")));
//!   [ns, sl, nr] }
//!
//! \\ (S) structure, for prime powers too large for (B). issquare(Mod(u,m),&r) is PARI's
//! \\     independent existence oracle (O) and gives one root r of the unit part; the others
//! \\     are -r (odd p) or -r, r + 2^(j-1), 2^(j-1) - r (p = 2, j >= 3).
//! unitroots(u,p,j) = { my(m=p^j, r); if(!issquare(Mod(u,m),&r), return([])); r = lift(r);
//!   if(p==2, if(j==1, return([1])); if(j==2, return([1,3]));
//!     return(vecsort([r, m-r, (r+m/2)%m, (m/2-r+m)%m],,8)));
//!   vecsort([r, m-r],,8) }
//! \\     [least root, #roots] of x^2 = a (mod p^k), from a = p^(2e)*u and x = p^e*y
//! lr(a,p,k) = { my(m=p^k, b=a%m, v, u, S); if(b==0, return([0, p^(k\2)]));
//!   v=valuation(b,p); if(v%2, return(["none",0]));
//!   u=b/p^v; S=unitroots(u,p,k-v); if(#S==0, return(["none",0]));
//!   [p^(v/2)*S[1], #S*p^(v/2)] }
//! \\     cross-check of (S): it agrees with issquare (O), and the least root squares back
//! chk(a,p,k) = { my(m=p^k, L=lr(a,p,k), s=issquare(Mod(a,m)));
//!   if((L[1]!="none") != s, error("structure vs issquare"));
//!   if(s && Mod(L[1],m)^2 != Mod(a,m), error("least root does not square back")); L }
//!
//! \\ (C) the composite rule of IntegerMod::sqrt (for squares a only): CRT of the
//! \\     per-prime-power least roots; the number of roots is the product of the counts
//! crt_least(a,m) = { my(F=factor(m)); lift(chinese(vector(#F~, i,
//!   Mod(lr(a,F[i,1],F[i,2])[1], F[i,1]^F[i,2])))) }
//! nroots(a,m) = { my(F=factor(m)); prod(i=1, #F~, lr(a,F[i,1],F[i,2])[2]) }
//! ```
//!
//! Source per test: `prime_power_review_cases`, `prime_power_non_squares`, `composite_moduli`
//! (root lists) and `handbook_h19e4` are (B), with (O) for the non-squares;
//! `prime_power_exhaustive_fingerprints` is (F) plus an in-test brute force;
//! `prime_power_large` and `composite_large` are (S) checked by `chk` (so (O) plus square-back),
//! plus (C) for the composite; the `sqrt` values of `composite_moduli` are (C);
//! `composite_exhaustive_small` is an in-test brute force. The modulus-0 sentinel tests are
//! plain arithmetic in Z.

use rustmath_finitefields::{
    all_square_roots_mod_prime_power, square_root_mod_prime_power, IntegerMod,
};
use rustmath_integers::Integer;

fn i(x: i64) -> Integer {
    Integer::from(x)
}

fn big(s: &str) -> Integer {
    Integer::from_decimal_str(s).expect("decimal literal")
}

fn squares_back(x: &Integer, a: &Integer, m: &Integer) -> bool {
    (x.clone() * x.clone() - a.clone()).modulo(m).is_zero()
}

fn zmod(a: &Integer, m: &Integer) -> IntegerMod {
    IntegerMod::new(a.clone(), m.clone()).expect("modulus > 1")
}

fn values(v: &[IntegerMod]) -> Vec<Integer> {
    v.iter().map(|r| r.value().clone()).collect()
}

/// Brute force in i64 (the in-test counterpart of gp `roots`).
fn brute_roots(a: i64, m: i64) -> Vec<i64> {
    (0..m).filter(|x| (x * x - a).rem_euclid(m) == 0).collect()
}

/// Review cases (ch018.Modsqrt, ch047.SquareRoot) and the odd-p, k >= 3 wrong-root
/// regressions. (a, p, k, least root, all roots); gp (B): `roots(a, p^k)`.
#[test]
fn prime_power_review_cases() {
    let cases: &[(i64, i64, u32, i64, &[i64])] = &[
        // Squares the old code called non-squares (false None).
        (4, 2, 3, 2, &[2, 6]),
        (9, 2, 4, 3, &[3, 5, 11, 13]),
        (17, 2, 5, 7, &[7, 9, 23, 25]),
        (9, 3, 3, 3, &[3, 6, 12, 15, 21, 24]),
        // The old code returned non-roots here: 22, 25, 18.
        (4, 3, 3, 2, &[2, 25]),
        (7, 3, 3, 13, &[13, 14]),
        (2, 7, 3, 108, &[108, 235]),
        (2, 7, 2, 10, &[10, 39]),
        // p = 2, all unit-root shapes, and p | a.
        (1, 2, 1, 1, &[1]),
        (1, 2, 2, 1, &[1, 3]),
        (1, 2, 3, 1, &[1, 3, 5, 7]),
        (1, 2, 5, 1, &[1, 15, 17, 31]),
        (33, 2, 6, 15, &[15, 17, 47, 49]),
        (64, 2, 7, 8, &[8, 24, 40, 56, 72, 88, 104, 120]),
        (
            16,
            2,
            10,
            4,
            &[
                4, 124, 132, 252, 260, 380, 388, 508, 516, 636, 644, 764, 772, 892, 900, 1020,
            ],
        ),
        (
            144,
            2,
            10,
            12,
            &[
                12, 116, 140, 244, 268, 372, 396, 500, 524, 628, 652, 756, 780, 884, 908, 1012,
            ],
        ),
        (63, 3, 5, 39, &[39, 42, 120, 123, 201, 204]),
        (
            81,
            3,
            5,
            9,
            &[
                9, 18, 36, 45, 63, 72, 90, 99, 117, 126, 144, 153, 171, 180, 198, 207, 225, 234,
            ],
        ),
        // Negative a is reduced first.
        (-1, 5, 3, 57, &[57, 68]),
        (-7, 2, 6, 11, &[11, 21, 43, 53]),
        // a = 0: the multiples of p^ceil(k/2).
        (0, 2, 1, 0, &[0]),
        (0, 2, 2, 0, &[0, 2]),
        (0, 2, 3, 0, &[0, 4]),
        (0, 2, 4, 0, &[0, 4, 8, 12]),
        (0, 3, 3, 0, &[0, 9, 18]),
        (0, 3, 5, 0, &[0, 27, 54, 81, 108, 135, 162, 189, 216]),
    ];
    for &(a, p, k, least, all) in cases {
        let (ai, pi) = (i(a), i(p));
        let pk = pi.pow(k);
        let all_exp: Vec<Integer> = all.iter().map(|&x| i(x)).collect();

        let got = square_root_mod_prime_power(&ai, &pi, k as usize);
        assert_eq!(got, Some(i(least)), "least root of {a} mod {p}^{k}");
        let all_got = all_square_roots_mod_prime_power(&ai, &pi, k as usize);
        assert_eq!(all_got, all_exp, "all roots of {a} mod {p}^{k}");
        for x in &all_got {
            assert!(squares_back(x, &ai, &pk));
        }

        // The same through IntegerMod (a prime-power modulus: sqrt is the least root).
        let x = zmod(&ai, &pk);
        assert!(x.is_square(), "{a} mod {p}^{k} is a square");
        assert_eq!(x.sqrt().map(|r| r.value().clone()), Some(i(least)));
        assert_eq!(values(&x.all_sqrts()), all_exp);
        assert_eq!(x.num_sqrts(), i(all.len() as i64));
    }
}

/// Provable non-squares; gp (O): `issquare(Mod(a, p^k))` = 0 and (B): `roots(a, p^k)` = [].
#[test]
fn prime_power_non_squares() {
    let cases: &[(i64, i64, u32)] = &[
        (8, 2, 4),   // odd valuation 3
        (2, 2, 3),   // odd valuation 1
        (12, 2, 4),  // 4*3, 3 != 1 mod 4
        (3, 2, 3),   // unit, not 1 mod 8
        (5, 2, 3),   // unit, not 1 mod 8
        (3, 2, 2),   // unit, not 1 mod 4
        (2, 2, 2),   // odd valuation 1
        (32, 2, 7),  // odd valuation 5
        (3, 3, 2),   // odd valuation 1
        (18, 3, 3),  // 9*2, 2 a non-residue mod 3
        (162, 3, 5), // 81*2
        (3, 7, 1),   // non-residue mod 7
    ];
    for &(a, p, k) in cases {
        let (ai, pi) = (i(a), i(p));
        assert_eq!(
            square_root_mod_prime_power(&ai, &pi, k as usize),
            None,
            "{a} mod {p}^{k}"
        );
        assert!(all_square_roots_mod_prime_power(&ai, &pi, k as usize).is_empty());
        let x = zmod(&ai, &pi.pow(k));
        assert!(!x.is_square());
        assert!(x.sqrt().is_none());
        assert!(x.all_sqrts().is_empty());
        assert_eq!(x.num_sqrts(), i(0));
    }
}

/// Exhaustive check per prime power: the least root and the full root list equal an in-test
/// brute force for every a in [0, p^k), and the per-p^k fingerprint
/// (#squares, sum of least roots) equals gp (F) `tab(p, k)`; the root counts sum to p^k.
#[test]
fn prime_power_exhaustive_fingerprints() {
    #[rustfmt::skip]
    let table: &[(i64, u32, usize, i64)] = &[
        (2, 1, 2, 1), (2, 2, 2, 1), (2, 3, 3, 3), (2, 4, 4, 6), (2, 5, 7, 22),
        (2, 6, 12, 76), (2, 7, 23, 300), (2, 8, 44, 1176),
        (3, 1, 2, 1), (3, 2, 4, 7), (3, 3, 11, 64), (3, 4, 31, 568), (3, 5, 92, 5113),
        (5, 1, 3, 3), (5, 2, 11, 63), (5, 3, 53, 1578),
        (7, 1, 4, 6), (7, 2, 22, 258), (7, 3, 151, 12648),
        (11, 2, 56, 1665), (13, 2, 79, 3297),
    ];
    for &(p, k, n_squares, sum_least) in table {
        let m = p.pow(k);
        let (mut n, mut s, mut total_roots) = (0usize, 0i64, 0usize);
        for a in 0..m {
            let brute = brute_roots(a, m);
            let got = square_root_mod_prime_power(&i(a), &i(p), k as usize);
            assert_eq!(got, brute.first().map(|&x| i(x)), "{a} mod {p}^{k}");
            let all = all_square_roots_mod_prime_power(&i(a), &i(p), k as usize);
            assert_eq!(
                all,
                brute.iter().map(|&x| i(x)).collect::<Vec<_>>(),
                "{a} mod {p}^{k}"
            );
            total_roots += all.len();
            if let Some(r) = got {
                n += 1;
                s += r.to_i64();
            }
        }
        assert_eq!((n, s), (n_squares, sum_least), "fingerprint of {p}^{k}");
        assert_eq!(total_roots as i64, m, "root counts of {p}^{k}");
    }
}

/// Large prime powers. gp (S) checked by `chk(a, p, k)`: existence from issquare (O), least
/// root and count from the structure, and the least root squares back.
#[test]
fn prime_power_large() {
    // P = 1000000000000000177: the first prime > 10^18 with P = 1 mod 16 (gp:
    // `p = nextprime(10^18); while(p % 16 != 1, p = nextprime(p+1)); p`); P = 17 mod 32, so
    // Tonelli-Shanks runs with s = 4.
    let p18 = big("1000000000000000177");
    // (a, p, k, least root, #roots); #roots = None where factoring p^k by trial division
    // (IntegerMod::num_sqrts) is infeasible (the integer_mod.rs unit tests cover those counts).
    type Case = (Integer, Integer, u32, Option<Integer>, Option<Integer>);
    let cases: Vec<Case> = vec![
        (
            i(17),
            i(2),
            100,
            Some(big("217788382556221841343574235415")),
            Some(i(4)),
        ),
        (
            i(2),
            i(7),
            30,
            Some(big("9738858034492596758892959")),
            Some(i(2)),
        ),
        // 2^20*41
        (
            big("42991616"),
            i(2),
            64,
            Some(big("4494817251505152")),
            Some(i(4096)),
        ),
        // 2^21*41 (odd valuation), 2^20*43 (unit part 43 = 3 mod 8)
        (big("85983232"), i(2), 64, None, Some(i(0))),
        (big("45088768"), i(2), 64, None, Some(i(0))),
        // 2^62: least root 2^31, 2^32 roots
        (
            big("4611686018427387904"),
            i(2),
            64,
            Some(big("2147483648")),
            Some(big("4294967296")),
        ),
        // 2^62*3, 2^63, 2^60*5: no root
        (big("13835058055282163712"), i(2), 64, None, Some(i(0))),
        (big("9223372036854775808"), i(2), 64, None, Some(i(0))),
        (big("5764607523034234880"), i(2), 64, None, Some(i(0))),
        (i(0), i(2), 64, Some(i(0)), Some(big("4294967296"))),
        // Unreduced input: 2^60*17 > 2^64 reduces to 2^60 mod 2^64; least root 2^30.
        (
            big("19599665578316398592"),
            i(2),
            64,
            Some(big("1073741824")),
            Some(big("4294967296")),
        ),
        // 3^10*7 and 3^10*2
        (i(413343), i(3), 25, Some(i(92959650)), Some(i(486))),
        (i(118098), i(3), 25, None, Some(i(0))),
        (i(-1), i(5), 20, Some(big("15613890344818")), Some(i(2))),
        (
            i(-7),
            i(2),
            70,
            Some(big("108828153058997780299")),
            Some(i(4)),
        ),
        // Modulo powers of P (also gp p-adic: sqrt(13 + O(P^k)), issquare(5 + O(P^3)) = 0).
        (i(5), p18.clone(), 3, None, None),
        (i(7), p18.clone(), 3, None, None),
        (i(11), p18.clone(), 3, None, None),
        (i(13), p18.clone(), 1, Some(big("222291380919551995")), None),
        (
            i(13),
            p18.clone(),
            2,
            Some(big("197726030944133057775216096191993076")),
            None,
        ),
        (
            i(13),
            p18.clone(),
            3,
            Some(big(
                "265840322487973224909748129798363644736247129518228023",
            )),
            None,
        ),
        // P^2*13 (2P roots), P*5 (odd valuation), P^2*5 (unit part 5 a non-residue)
        (
            big("13000000000000004602000000000000407277"),
            p18.clone(),
            3,
            Some(big("222291380919552034345574422760703115")),
            None,
        ),
        (big("5000000000000000885"), p18.clone(), 3, None, None),
        (
            big("5000000000000001770000000000000156645"),
            p18.clone(),
            3,
            None,
            None,
        ),
    ];
    for (a, p, k, least, count) in cases {
        let pk = p.pow(k);
        let got = square_root_mod_prime_power(&a, &p, k as usize);
        assert_eq!(got, least, "least root of {a} mod {p}^{k}");
        if let Some(x) = &got {
            assert!(squares_back(x, &a, &pk), "{x}^2 = {a} mod {p}^{k}");
        }
        if let Some(count) = count {
            let x = zmod(&a, &pk);
            assert_eq!(x.num_sqrts(), count, "count of roots of {a} mod {p}^{k}");
            assert_eq!(x.sqrt().map(|r| r.value().clone()), least);
            assert_eq!(x.is_square(), least.is_some());
        }
    }
    // Small enough to list: 2^20*41 mod 2^64 has 4096 roots, the least first, all distinct.
    let a = big("42991616");
    let m = i(2).pow(64);
    let all = all_square_roots_mod_prime_power(&a, &i(2), 64);
    assert_eq!(all.len(), 4096);
    assert_eq!(all[0], big("4494817251505152"));
    assert!(all.windows(2).all(|w| w[0] < w[1]));
    assert!(all.iter().all(|x| squares_back(x, &a, &m) && *x < m));
}

/// k = 0 is Z/1Z = {0}: the residue 0 is a root (the old code returned None).
#[test]
fn prime_power_k_zero() {
    assert_eq!(square_root_mod_prime_power(&i(5), &i(3), 0), Some(i(0)));
    assert_eq!(
        all_square_roots_mod_prime_power(&i(5), &i(3), 0),
        vec![i(0)]
    );
}

#[test]
#[should_panic(expected = "not prime")]
fn prime_power_rejects_composite_p() {
    let _ = square_root_mod_prime_power(&i(4), &i(15), 2);
}

#[test]
#[should_panic(expected = "not prime")]
fn all_roots_prime_power_rejects_composite_p() {
    let _ = all_square_roots_mod_prime_power(&i(4), &i(15), 2);
}

/// Composite moduli: `sqrt` is the CRT of the per-prime-power least roots, gp (C)
/// `crt_least(a, m)`; the root lists are gp (B) `roots(a, m)`; non-squares have
/// `issquare(Mod(a, m))` = 0.
#[test]
fn composite_moduli() {
    let cases: &[(i64, i64, Option<i64>, &[i64])] = &[
        (36, 72, Some(18), &[6, 18, 30, 42, 54, 66]),
        (4, 15, Some(7), &[2, 7, 8, 13]),
        (4, 216, Some(2), &[2, 106, 110, 214]),
        (4, 24, Some(10), &[2, 10, 14, 22]),
        (12, 24, Some(18), &[6, 18]),
        (9, 54, Some(3), &[3, 15, 21, 33, 39, 51]),
        (36, 40, Some(26), &[6, 14, 26, 34]),
        (2, 49, Some(10), &[10, 39]),
        (17, 1568, None, &[]),
        (2, 15, None, &[]),
    ];
    for &(a, m, s, all) in cases {
        let x = zmod(&i(a), &i(m));
        assert_eq!(
            x.sqrt().map(|r| r.value().clone()),
            s.map(i),
            "sqrt {a} mod {m}"
        );
        assert_eq!(x.is_square(), s.is_some());
        let all_exp: Vec<Integer> = all.iter().map(|&v| i(v)).collect();
        assert_eq!(values(&x.all_sqrts()), all_exp, "all roots of {a} mod {m}");
        assert_eq!(x.num_sqrts(), i(all.len() as i64), "count {a} mod {m}");
        // Every listed root is an element of the same ring.
        assert!(x.all_sqrts().iter().all(|r| r.modulus() == &i(m)));
    }
    // 9 mod 432 = 2^4*3^3: gp crt_least(9, 432) = 3, nroots(9, 432) = 24 = #roots(9, 432).
    let x = zmod(&i(9), &i(432));
    assert_eq!(x.sqrt().map(|r| r.value().clone()), Some(i(3)));
    assert_eq!(x.num_sqrts(), i(24));
    assert_eq!(
        values(&x.all_sqrts()),
        brute_roots(9, 432).into_iter().map(i).collect::<Vec<_>>()
    );
}

/// Handbook example H19E4 (ch. 19): the 12 square roots of 1404 in Z/2340Z, gp (B)
/// `roots(1404, 2340)`. `sqrt` returns the CRT of the least roots mod 4, 9, 5, 13 (gp (C)
/// `crt_least(1404, 2340)` = 1872), not the least root 78.
#[test]
fn handbook_h19e4() {
    let x = zmod(&i(1404), &i(2340));
    let expected = [
        78, 312, 468, 702, 858, 1092, 1248, 1482, 1638, 1872, 2028, 2262,
    ]
    .map(i);
    assert_eq!(values(&x.all_sqrts()), expected.to_vec());
    assert_eq!(x.num_sqrts(), i(12));
    assert_eq!(x.sqrt().map(|r| r.value().clone()), Some(i(1872)));
    assert!(x.is_square());
    for r in x.all_sqrts() {
        assert_eq!(r.clone() * r, x);
    }
}

/// A composite modulus with large prime powers: m = 2^10*3^7*7^5*101^3, a = 123456789^2 mod m.
/// gp (S): `[lr(a,2,10), lr(a,3,7), lr(a,7,5), lr(a,101,3)]` =
/// [[235, 4], [90, 18], [7433, 2], [179331, 2]]; (C) `crt_least(a, m)` = 6463262613543147
/// (squares back), `nroots(a, m)` = 288.
#[test]
fn composite_large() {
    let m = big("38779576421999616");
    assert_eq!(m, i(2).pow(10) * i(3).pow(7) * i(7).pow(5) * i(101).pow(3));
    let a = big("15241578750190521");
    assert_eq!(a, (i(123456789) * i(123456789)).modulo(&m));
    let x = zmod(&a, &m);
    assert_eq!(
        x.sqrt().map(|r| r.value().clone()),
        Some(big("6463262613543147"))
    );
    assert_eq!(x.num_sqrts(), i(288));
    let all = values(&x.all_sqrts());
    assert_eq!(all.len(), 288);
    assert!(all.windows(2).all(|w| w[0] < w[1]));
    assert!(all.iter().all(|r| squares_back(r, &a, &m)));
    assert!(all.contains(&big("6463262613543147")));
    assert!(all.contains(&i(123456789)));
    // The per-prime-power least roots, through the free function (gp (S) values above).
    for (p, k, least) in [(2, 10, 235), (3, 7, 90), (7, 5, 7433), (101, 3, 179331)] {
        assert_eq!(square_root_mod_prime_power(&a, &i(p), k), Some(i(least)));
    }
}

/// Every m <= 200 and every a in [0, m), against an in-test brute force: `sqrt` exists iff a
/// root exists, squares back, and is the least root when m is a prime power; `all_sqrts` and
/// `num_sqrts` equal the brute-force list and its length.
#[test]
fn composite_exhaustive_small() {
    for m in 2i64..=200 {
        let prime_power = rustmath_integers::prime::factor(&i(m)).len() == 1;
        for a in 0..m {
            let brute = brute_roots(a, m);
            let x = zmod(&i(a), &i(m));
            let s = x.sqrt().map(|r| r.value().clone());
            assert_eq!(s.is_some(), !brute.is_empty(), "{a} mod {m}");
            assert_eq!(x.is_square(), !brute.is_empty(), "{a} mod {m}");
            if let Some(r) = &s {
                assert!(squares_back(r, &i(a), &i(m)), "{r}^2 = {a} mod {m}");
                if prime_power {
                    assert_eq!(*r, i(brute[0]), "least root of {a} mod {m}");
                }
            }
            let all = values(&x.all_sqrts());
            assert_eq!(
                all,
                brute.iter().map(|&v| i(v)).collect::<Vec<_>>(),
                "{a} mod {m}"
            );
            assert_eq!(x.num_sqrts(), i(brute.len() as i64), "{a} mod {m}");
        }
    }
}

/// Negative and unreduced inputs to the free function equal the reduced ones (every a + s*p^k
/// for s in {-3, -1, 1, 2}).
#[test]
fn prime_power_unreduced_inputs() {
    for (p, k) in [(2i64, 6u32), (3, 4), (5, 3)] {
        let m = p.pow(k);
        for a in 0..m {
            let expected = square_root_mod_prime_power(&i(a), &i(p), k as usize);
            let all = all_square_roots_mod_prime_power(&i(a), &i(p), k as usize);
            for s in [-3i64, -1, 1, 2] {
                let b = i(a + s * m);
                assert_eq!(square_root_mod_prime_power(&b, &i(p), k as usize), expected);
                assert_eq!(all_square_roots_mod_prime_power(&b, &i(p), k as usize), all);
            }
        }
    }
}

/// The modulus-0 sentinel is an element of Z (Z/0Z = Z): perfect squares only (no gp needed).
#[test]
fn sentinel_square_roots_in_z() {
    use rustmath_core::Ring;
    let one = <IntegerMod as Ring>::one();
    let zero = <IntegerMod as Ring>::zero();
    let two = one.clone() + one.clone();
    let four = two.clone() * two.clone();

    assert_eq!(values(&four.all_sqrts()), vec![i(-2), i(2)]);
    assert_eq!(four.num_sqrts(), i(2));
    assert!(four.is_square());
    assert_eq!(four.sqrt().map(|r| r.value().clone()), Some(i(2)));

    assert_eq!(values(&zero.all_sqrts()), vec![i(0)]);
    assert_eq!(zero.num_sqrts(), i(1));
    assert!(zero.is_square());

    assert!(two.all_sqrts().is_empty());
    assert_eq!(two.num_sqrts(), i(0));
    assert!(!two.is_square());

    let minus_four = -four;
    assert!(minus_four.all_sqrts().is_empty());
    assert_eq!(minus_four.num_sqrts(), i(0));
    assert!(minus_four.all_sqrts().iter().all(|r| r.modulus().is_zero()));
}
