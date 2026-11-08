//! Squares modulo a prime: `PrimeField::is_quadratic_residue` and `square_root_mod_prime`.
//!
//! MAGMA Handbook (digests in `rustmath-lava/docs/`): ch. 21 §21.4.4 `IsSquare(a)` and §21.4.7
//! "Order and Roots" `SquareRoot(a)` / `Sqrt(a)` (finite fields); ch. 19 §19.5.6 `IsSquare`
//! (Z/mZ); ch. 18 §18.12.1 `Modsqrt(n, m)` (for a prime m). Algorithms: Jacobi symbol by
//! quadratic reciprocity and Tonelli–Shanks (H. Cohen, *A Course in Computational Algebraic
//! Number Theory*, GTM 138, Algorithms 1.4.10 and 1.5.1), both in `rustmath-integers`.
//!
//! Regressions:
//! - `PrimeField::is_quadratic_residue` said that 1 is not a square in GF(2): it turned the
//!   `Err` of the Legendre symbol (defined for odd p only) into `false`.
//! - `square_root_mod_prime` turned a composite p into a false `None` (4 mod 15, 4 mod 9), and
//!   a strong pseudoprime p into a false `None` (73 mod ψ12). It now refuses a modulus that
//!   `is_prime` rejects (documented panic, as `square_root_mod_prime_power` since b4ded72), and
//!   `None` is always a Jacobi-symbol proof.
//!
//! # Reference values: PARI/GP 2.15.4 (`gp -q`)
//!
//! ```text
//! \\ (A) squares in GF(p)
//! forprime(p=2, 13, print("p=",p," issquare=", vector(p, a, issquare(Mod(a-1,p)))))
//! \\ (F) fingerprint over the primes below 200: [#squares, sum of the square residues]
//! my(S=0, C=0); forprime(p=2,199, my(v=select(a->issquare(Mod(a,p)), [0..p-1]));
//!   C+=#v; S+=vecsum(v)); [C, S]                                 \\ [2137, 133894]
//! \\ (L) large primes 2^61-1, 10^18+9, 2^89-1, 2^127-1 and a in [-1,2,3,5,7,10,11]
//! vector(7, j, issquare(Mod([-1,2,3,5,7,10,11][j], p)))
//! \\ (R) the root set of a mod p, or "none"
//! rts(a,p) = { my(r); if(!issquare(Mod(a,p),&r), return("none")); r=lift(r);
//!   vecsort([r,(p-r)%p],,8) }
//! \\ (P) psi12 = 318665857834031151167461, the least strong pseudoprime to the bases 2..37
//! \\     (Sorenson-Webster 2017): factor(psi12) = [399165290221, 1; 798330580441, 1];
//! \\     kronecker(a, psi12) and issquare(Mod(a, psi12)) for a = 2, 4, 41, 43, 73
//! \\     the four roots of 4: vecsort(concat(vector(2,i,vector(2,j,
//! \\       lift(chinese(Mod((-1)^i*2,q), Mod((-1)^j*2,r)))))))  with q, r the two factors
//! ```

use rustmath_core::Ring;
use rustmath_finitefields::{square_root_mod_prime, square_root_mod_prime_power, PrimeField};
use rustmath_integers::Integer;
use std::panic::catch_unwind;

fn i(x: i64) -> Integer {
    Integer::from(x)
}

fn big(s: &str) -> Integer {
    Integer::from_decimal_str(s).expect("decimal literal")
}

fn gfp(a: i64, p: i64) -> PrimeField {
    PrimeField::new(i(a), i(p)).expect("modulus > 1")
}

fn squares_back(x: &Integer, a: &Integer, m: &Integer) -> bool {
    (x.clone() * x.clone() - a.clone()).modulo(m).is_zero()
}

fn small_primes_below(n: i64) -> Vec<i64> {
    (2..n)
        .filter(|&p| (2..p).take_while(|d| d * d <= p).all(|d| p % d != 0))
        .collect()
}

/// The set {x^2 mod p}, as a membership table (in-test brute force).
fn square_table(p: i64) -> Vec<bool> {
    let mut t = vec![false; p as usize];
    for x in 0..p {
        t[((x * x) % p) as usize] = true;
    }
    t
}

/// ψ12 = 399165290221 · 798330580441 (PARI `factor`), a strong pseudoprime to the bases 2..37.
const PSI12: &str = "318665857834031151167461";

// ---------------------------------------------------------------------------------------
// (a) PrimeField::is_quadratic_residue
// ---------------------------------------------------------------------------------------

// Regression: 1 = 1^2 in GF(2). PARI `issquare(Mod(1,2))` = 1, `issquare(Mod(0,2))` = 1.
#[test]
fn is_quadratic_residue_gf2() {
    assert!(gfp(0, 2).is_quadratic_residue());
    assert!(gfp(1, 2).is_quadratic_residue());
    // 3 reduces to 1, -1 reduces to 1.
    assert!(gfp(3, 2).is_quadratic_residue());
    assert!(gfp(-1, 2).is_quadratic_residue());
}

// PARI (A): the six vectors, element a-1 at index a.
#[test]
fn is_quadratic_residue_small_primes_match_pari() {
    let pari: &[(i64, &[u8])] = &[
        (2, &[1, 1]),
        (3, &[1, 1, 0]),
        (5, &[1, 1, 0, 0, 1]),
        (7, &[1, 1, 1, 0, 1, 0, 0]),
        (11, &[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
        (13, &[1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1]),
    ];
    for &(p, expected) in pari {
        for a in 0..p {
            assert_eq!(
                gfp(a, p).is_quadratic_residue(),
                expected[a as usize] == 1,
                "is_quadratic_residue({a} in GF({p}))"
            );
        }
    }
}

// Every prime p < 200 and every a in [0, p): in-test brute force, plus the PARI fingerprint
// (F): 2137 squares in total, the square residues sum to 133894.
#[test]
fn is_quadratic_residue_primes_below_200() {
    let mut count = 0u64;
    let mut sum = 0i64;
    for p in small_primes_below(200) {
        let table = square_table(p);
        for a in 0..p {
            let qr = gfp(a, p).is_quadratic_residue();
            assert_eq!(
                qr, table[a as usize],
                "is_quadratic_residue({a} in GF({p}))"
            );
            if qr {
                count += 1;
                sum += a;
            }
        }
    }
    assert_eq!((count, sum), (2137, 133894));
}

// PARI (L). 2^89-1 and 2^127-1 lie above ψ12, where `is_prime` is only a probable-prime test;
// the answers are still decisions (a Jacobi symbol -1, or a root that squares back).
#[test]
fn is_quadratic_residue_large_primes_match_pari() {
    let a_values = [-1i64, 2, 3, 5, 7, 10, 11];
    let table: &[(&str, [u8; 7])] = &[
        ("2305843009213693951", [0, 1, 0, 1, 0, 1, 0]),
        ("1000000000000000009", [1, 1, 1, 1, 0, 1, 0]),
        ("618970019642690137449562111", [0, 1, 0, 1, 1, 1, 0]),
        (
            "170141183460469231731687303715884105727",
            [0, 1, 0, 0, 0, 0, 1],
        ),
    ];
    for (p, expected) in table {
        let p = big(p);
        for (a, e) in a_values.iter().zip(expected.iter()) {
            let x = PrimeField::new(i(*a), p.clone()).unwrap();
            assert_eq!(x.is_quadratic_residue(), *e == 1, "issquare(Mod({a}, {p}))");
        }
    }
}

// The modulus-0 sentinel is an element of Z: a perfect square or not (unchanged behaviour).
#[test]
fn is_quadratic_residue_sentinel_unchanged() {
    let one = <PrimeField as Ring>::one();
    assert!(one.is_quadratic_residue());
    assert!(!(one.clone() + one.clone()).is_quadratic_residue()); // 2 is not a square in Z
    assert!(!(-one.clone()).is_quadratic_residue()); // -1 is not a square in Z
    let four = (one.clone() + one.clone()) * (one.clone() + one);
    assert!(four.is_quadratic_residue());
    assert!(<PrimeField as Ring>::zero().is_quadratic_residue());
}

// 0 is a square for every modulus; no primality check is needed (and none panics).
#[test]
fn is_quadratic_residue_zero_never_panics() {
    assert!(gfp(0, 15).is_quadratic_residue());
    assert!(gfp(15, 15).is_quadratic_residue());
}

// PrimeField::new does not check primality. PARI `issquare(Mod(4,15))` = 1, so the old `false`
// was wrong; GF(15) does not exist, and the function now refuses.
#[test]
#[should_panic(expected = "not prime")]
fn is_quadratic_residue_rejects_composite_modulus() {
    let _ = gfp(4, 15).is_quadratic_residue();
}

// ---------------------------------------------------------------------------------------
// (b) square_root_mod_prime
// ---------------------------------------------------------------------------------------

// PARI (R): every result lies in the PARI root set (and squares back); `None` exactly for the
// "none" rows. Includes the p = 2 trap: kronecker(3, 2) = kronecker(-3, 2) = -1 for the
// unreduced value, but 3 ≡ -3 ≡ 1 is a square mod 2 (root 1).
#[test]
fn square_root_mod_prime_matches_pari() {
    let cases: &[(&str, &str, Option<[&str; 2]>)] = &[
        ("4", "7", Some(["2", "5"])),
        ("3", "7", None),
        ("1", "7", Some(["1", "6"])),
        ("0", "7", Some(["0", "0"])),
        ("4", "11", Some(["2", "9"])),
        ("-1", "5", Some(["2", "3"])),
        ("-1", "7", None),
        ("3", "2", Some(["1", "1"])),
        ("-3", "2", Some(["1", "1"])),
        ("0", "2", Some(["0", "0"])),
        ("5", "13", None),
        ("2", "17", Some(["6", "11"])),
        ("10", "13", Some(["6", "7"])),
        (
            "2",
            "2305843009213693951",
            Some(["2147483648", "2305843007066210303"]),
        ),
        (
            "-1",
            "1000000000000000009",
            Some(["333333333000000003", "666666667000000006"]),
        ),
        (
            "3",
            "1000000000000000009",
            Some(["60248662633209561", "939751337366790448"]),
        ),
        ("7", "1000000000000000009", None),
        (
            "2",
            "170141183460469231731687303715884105727",
            Some([
                "18446744073709551616",
                "170141183460469231713240559642174554111",
            ]),
        ),
        (
            "11",
            "170141183460469231731687303715884105727",
            Some([
                "18756936442469208154408315811770624301",
                "151384247018000023577278987904113481426",
            ]),
        ),
    ];
    for (a, p, roots) in cases {
        let (a, p) = (big(a), big(p));
        let got = square_root_mod_prime(&a, &p);
        match roots {
            None => assert_eq!(got, None, "sqrt({a} mod {p})"),
            Some(rs) => {
                let r = got.unwrap_or_else(|| panic!("sqrt({a} mod {p}): false None"));
                assert!(rs.iter().any(|s| big(s) == r), "sqrt({a} mod {p}) = {r}");
                assert!(squares_back(&r, &a, &p), "sqrt({a} mod {p}) = {r}");
                assert!(r >= i(0) && r < p, "sqrt({a} mod {p}) = {r} not reduced");
            }
        }
    }
}

// Every prime p < 200 and a in [-3, p + 3): `None` iff a mod p is not a square (in-test brute
// force), otherwise a root in [0, p) that squares back.
#[test]
fn square_root_mod_prime_exhaustive_small_primes() {
    for p in small_primes_below(200) {
        let table = square_table(p);
        for a in -3..p + 3 {
            let got = square_root_mod_prime(&i(a), &i(p));
            let is_square = table[a.rem_euclid(p) as usize];
            match got {
                None => assert!(!is_square, "false None for {a} mod {p}"),
                Some(r) => {
                    assert!(is_square, "root {r} for the non-square {a} mod {p}");
                    assert!(r >= i(0) && r < i(p) && squares_back(&r, &i(a), &i(p)));
                }
            }
        }
    }
}

// PARI: issquare(Mod(4,15)) = 1 (roots 2, 7, 8, 13), so the old `None` was false. A composite
// modulus is refused (PARI too: sqrt(Mod(4,15)) is "not a prime number in sqrt").
#[test]
#[should_panic(expected = "p = 15 is not prime")]
fn square_root_mod_prime_rejects_composite_p() {
    let _ = square_root_mod_prime(&i(4), &i(15));
}

// 4 is a square mod 9 and mod 25 (PARI issquare = 1); 1, 0 and -7 are not primes.
#[test]
fn square_root_mod_prime_rejects_non_primes() {
    for p in [9i64, 25, 1, 0, -7] {
        let r = catch_unwind(|| square_root_mod_prime(&i(4), &i(p)));
        assert!(
            r.is_err(),
            "square_root_mod_prime(4, {p}) must panic, got {r:?}"
        );
    }
}

// ψ12 passes `is_prime`. PARI (P): kronecker(2, ψ12) = kronecker(41, ψ12) = -1 and
// issquare = 0; 4 has the four roots [2, 3193322321766, 318665857830837828845695,
// 318665857834031151167459]; 43 has kronecker +1 and issquare 0; 73 has kronecker +1 and
// issquare 1 (the old code gave a false `None` for 73).
#[test]
fn square_root_mod_prime_pseudoprime_never_false_none() {
    let psi = big(PSI12);
    assert!(rustmath_integers::prime::is_prime(&psi));
    // Jacobi symbol -1: a proof for every odd modulus.
    assert_eq!(square_root_mod_prime(&i(2), &psi), None);
    assert_eq!(square_root_mod_prime(&i(41), &psi), None);
    // A square with Jacobi +1: a verified root.
    let r = square_root_mod_prime(&i(4), &psi).expect("4 = 2^2");
    let pari_roots = [
        "2",
        "3193322321766",
        "318665857830837828845695",
        "318665857834031151167459",
    ];
    assert!(pari_roots.iter().any(|s| big(s) == r), "root {r}");
    assert!(squares_back(&r, &i(4), &psi));
    // Tonelli-Shanks proves ψ12 composite: a panic, never a (false) `None`.
    for a in [73i64, 43] {
        let res = catch_unwind(|| square_root_mod_prime(&i(a), &psi));
        assert!(res.is_err(), "square_root_mod_prime({a}, psi12) = {res:?}");
    }
    // The prime-power function is outside its contract for ψ12 (result unspecified), but it
    // no longer gives the false `None` for 73: either a panic or a root that squares back.
    match catch_unwind(|| square_root_mod_prime_power(&i(73), &psi, 1)) {
        Err(_) => {}
        Ok(Some(r)) => assert!(squares_back(&r, &i(73), &psi)),
        Ok(None) => panic!("square_root_mod_prime_power(73, psi12, 1): false None"),
    }
}
