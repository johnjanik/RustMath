//! Primality tests and proofs of `rustmath-integers` against PARI/GP 2.15.4.
//!
//! MAGMA Handbook ch. 18 §18.9 (digest `rustmath-lava/docs/Chapter_18_Ring_of_Integers.md`):
//! `IsPrime(n)` (`Proof := true`, ECPP [AM93] for large n), `IsProbablePrime`,
//! `PrimalityCertificate` / `IsPrimeCertificate`, `IsPrimePower`, `NextPrime`, `PreviousPrime`,
//! `PrimesInInterval`, `NthPrime`, `RandomPrime`, and the handbook examples H18E4 and H18E6.
//!
//! References: [AM93] A. O. L. Atkin, F. Morain, *Elliptic curves and primality proving*, Math.
//! Comp. 61 (1993) 29-68; R. Baillie, S. S. Wagstaff Jr., *Lucas pseudoprimes*, Math. Comp. 35
//! (1980) 1391-1417; C. Pomerance, J. L. Selfridge, S. S. Wagstaff Jr., *The pseudoprimes to
//! 25·10^9*, Math. Comp. 35 (1980) 1003-1026; R. Baillie, A. Fiori, S. S. Wagstaff Jr.,
//! *Strengthening the Baillie-PSW primality test*, Math. Comp. 90 (2021) 1931-1955;
//! J. Sorenson, J. Webster, *Strong pseudoprimes to twelve prime bases*, Math. Comp. 86 (2017)
//! 985-1003.
//!
//! Every expected value has its gp command next to it. Commands that build certificates are
//! run after `default(parisizemax, 2^30)`. The certificates C127, C40, C65 and C200 are the
//! output of gp `primecert`, and `primecertisvalid` accepts each.

use rustmath_integers::prime::{
    is_prime, is_prime_miller_rabin, is_probable_prime, next_prime, nth_prime, pollard_p_minus_1,
    previous_prime, prime_pi, prime_power, prime_range, prove_primality, random_prime,
    try_is_prime, try_next_prime, try_previous_prime, try_prime_range, try_random_prime,
    EcppCertificate, NotPrimeWitness, Primality,
};
use rustmath_integers::Integer;

fn z(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap_or_else(|| panic!("bad integer {s:?}"))
}

fn pow2(k: u32) -> Integer {
    Integer::from(2).pow(k)
}

fn i(v: i64) -> Integer {
    Integer::from(v)
}

/// psi_12 = 399165290221 * 798330580441 and psi_13 = 1287836182261 * 2575672364521: strong
/// pseudoprimes to the bases 2..37 (Sorenson-Webster 2017).
const PSI_12: &str = "318665857834031151167461";
const PSI_13: &str = "3317044064679887385961981";

/// gp (2.15.4, after default(parisizemax, 2^30)): primecert(2^127-1)
const C127: &str = "[[170141183460469231731687303715884105727, 6463158235835222539, 37711126033, 0, [123106756171564943260470175830543720431, 75271964349985580273014815876123588329]], [4511697245836234712605197733, -103300761668182, 206599692, 0, [2028653280926922362829473336, 351806756186690237552618680]], [21837870144725762773, -277408667, 408201703, 0, [562706585515371056, 10173121340883365254]]]";
/// gp (2.15.4, after default(parisizemax, 2^30)): primecert(nextprime(10^40))
const C40: &str = "[[10000000000000000000000000000000000000121, -22, 9459152, 1, [8986749531375339305571207626880980158008, 264228802877425555654626166509072893057]], [1057177218422962227480856634928797, 62077446265796342, 8, 1057177218422962227480856634928796, [905038879671103388539143088846700, 503347634661221293400802691838893]], [132147152302870270675426296141557, 19090413001043459, 3028746964909, 132147152237868558693369200973557, [66474910464743596764054967879224, 65282052016992429384368617117346]], [43630964829325274911, -13164857771, 1568796679, 0, [37072770371681123870, 18457839869214529154]]]";
/// gp (2.15.4, after default(parisizemax, 2^30)): primecert(nextprime(2^64))
const C65: &str =
    "[[18446744073709551629, -8423788454, 160388, 1, [10598342506117936052, 2225259013356795550]]]";
/// gp (2.15.4, after default(parisizemax, 2^30)): primecert(nextprime(2^200)), printed without spaces
const C200: &str = "[[1606938044258990275541962092341162602522202993782792835301611,-2391022397824249511879151920012,8,1606938044258990275541962092341162602522202993782792783633736,[704831506213111899350223130073372827186053703514081245307123,404447163700230180533850878479402991320758648779996743710079]],[200867255532373784442745261542944203115003405411833998402703,729312882178055564668625338015,3266899963693,0,[105569034479845235998700590018764131782294197385713236523750,9617524876792764652196213347170327419900190930342074369987]],[61485585039251559724157960282597555545959410773,-248832504787480421075750,113314756,0,[42654775544463006678774074528637931942266581861,21952464002273592984170274463851940271418412469]],[542608811152993699462744367689432636879,-34598500748955144479,7747,0,[74698642846912997559653832607325416980,95167258040923132667293324614560355463]],[70041152853103614237426470690381797,-506634500288363039,432672283,0,[62274972521381567150673448874101638,4585298551171428045709137699508281]],[161880378302632375330732639,23183140155083,2931909163617,0,[59677471860376618974051590,5420422437487908289826427]]]";
/// C200 with a line break in the middle of a number and an indented continuation (gp wraps
/// long output like this); `from_pari` ignores all ASCII whitespace.
const C200_WRAPPED: &str = "[[1606938044258990275541962092341162602522202993782792835301611,-2391022397824249511879151920012,8,1606938044258990275541962092341162602522202993782792783633736,[704831506213111899350223130073372827186053703514081245307123,404447163700230180533850878479402991320758648779996743710079]],[200867255532373784442745261542944203115003405411833998402703,729312882178055564668625338015,3266899963693,0,[105569034479845235998700590018764131782294197385713236523750,9617524876792764652196213347170327419900190
        \t930342074369987]],[61485585039251559724157960282597555545959410773,-248832504787480421075750,113314756,0,[42654775544463006678774074528637931942266581861,21952464002273592984170274463851940271418412469]],[542608811152993699462744367689432636879,-34598500748955144479,7747,0,[74698642846912997559653832607325416980,95167258040923132667293324614560355463]],[70041152853103614237426470690381797,-506634500288363039,432672283,0,[62274972521381567150673448874101638,4585298551171428045709137699508281]],[161880378302632375330732639,23183140155083,2931909163617,0,[59677471860376618974051590,5420422437487908289826427]]]";

#[test]
fn is_prime_rejects_strong_pseudoprimes_psi12_psi13() {
    // gp: isprime(318665857834031151167461) = 0,
    // factor(318665857834031151167461) = [399165290221, 1; 798330580441, 1];
    // factor(3317044064679887385961981) = [1287836182261, 1; 2575672364521, 1]
    for s in [PSI_12, PSI_13] {
        let n = z(s);
        assert!(!is_prime(&n), "is_prime({s})");
        assert!(!is_probable_prime(&n), "is_probable_prime({s})");
        assert_eq!(try_is_prime(&n), Ok(false), "try_is_prime({s})");
        // They pass the fixed-base test to the twelve bases 2..37.
        assert!(
            is_prime_miller_rabin(&n, 12),
            "{s} is a strong pseudoprime to 2..37"
        );
    }
    assert_eq!(z("399165290221") * z("798330580441"), z(PSI_12));
    assert_eq!(z("1287836182261") * z("2575672364521"), z(PSI_13));
}

#[test]
fn prove_primality_witnesses() {
    let not_prime = |n: &Integer| match prove_primality(n) {
        Primality::NotPrime(w) => {
            assert!(w.confirms(n), "{w:?} confirms {n}");
            w
        }
        other => panic!("prove_primality({n}) = {other:?}"),
    };
    // Selfridge's D for psi_12 and psi_13 is -7 (gp: the first D in 5, -7, 9, ... with
    // kronecker(D, n) == -1); both pass the base-2 strong test and fail the Lucas test.
    assert_eq!(not_prime(&z(PSI_12)), NotPrimeWitness::StrongLucas(-7));
    assert_eq!(not_prime(&z(PSI_13)), NotPrimeWitness::StrongLucas(-7));
    // gp: factor(561) = [3,1; 11,1; 17,1], factor(2047) = [23,1; 89,1]
    assert_eq!(not_prime(&i(561)), NotPrimeWitness::Divisor(i(3)));
    assert_eq!(not_prime(&i(2047)), NotPrimeWitness::Divisor(i(23)));
    // gp: factor(1000036000099) = [1000003,1; 1000033,1] (no factor below 1000);
    // Mod(2, 1000036000099)^1000036000098 != 1, so 2 is a strong witness.
    assert_eq!(
        not_prime(&i(1_000_036_000_099)),
        NotPrimeWitness::MillerRabin(2)
    );
    for n in [0i64, 1, -7] {
        assert_eq!(not_prime(&i(n)), NotPrimeWitness::BelowTwo);
    }
    assert_eq!(not_prime(&pow2(90)), NotPrimeWitness::Divisor(i(2)));
    // No witness confirms a prime (gp: isprime(2^127-1) = 1).
    let m127 = pow2(127) - Integer::one();
    for w in [
        NotPrimeWitness::MillerRabin(2),
        NotPrimeWitness::StrongLucas(-7),
        NotPrimeWitness::Divisor(i(3)),
        NotPrimeWitness::Square,
        NotPrimeWitness::BelowTwo,
    ] {
        assert!(!w.confirms(&m127), "{w:?} must not confirm 2^127-1");
    }
}

/// The 2^521 - 1 proof is about 5 s in a debug build (25 levels plus the verification).
#[test]
fn is_prime_mersenne_and_neighbours() {
    // gp: [isprime(2^89-1), isprime(2^107-1), isprime(2^127-1), isprime(2^521-1),
    //      isprime(2^89+1), isprime(2^127+1)] = [1, 1, 1, 1, 0, 0]
    for k in [89u32, 107, 127, 521] {
        assert!(is_prime(&(pow2(k) - Integer::one())), "2^{k} - 1");
    }
    for k in [89u32, 127] {
        assert!(!is_prime(&(pow2(k) + Integer::one())), "2^{k} + 1");
    }
}

#[test]
fn is_probable_prime_and_is_prime_agree_with_pari() {
    // gp: vector(6, i, k = [64,80,100,128,160,200][i]; nextprime(2^k) - 2^k)
    //     = [13, 13, 277, 51, 7, 235];
    // [isprime(nextprime(2^k)+2), isprime(2^k+1)] = [0, 0] for each k
    for (k, d) in [
        (64u32, 13i64),
        (80, 13),
        (100, 277),
        (128, 51),
        (160, 7),
        (200, 235),
    ] {
        let p = pow2(k) + i(d);
        assert!(is_probable_prime(&p), "2^{k} + {d}");
        assert!(is_prime(&p), "2^{k} + {d}");
        for c in [p.clone() + i(2), pow2(k) + Integer::one()] {
            assert!(!is_probable_prime(&c), "{c}");
            assert!(!is_prime(&c), "{c}");
        }
    }
    // gp: precprime(10^30) == 10^30 - 11
    let p = Integer::from(10).pow(30) - i(11);
    assert!(is_probable_prime(&p) && is_prime(&p));
}

/// gp (2.15.4, after default(parisizemax, 2^30)): primecertisvalid(C) = 1 for each constant.
#[test]
fn ecpp_accepts_pari_certificates() {
    for (name, text, n) in [
        ("C127", C127, pow2(127) - Integer::one()),
        ("C40", C40, Integer::from(10).pow(40) + i(121)),
        ("C65", C65, z("18446744073709551629")),
        ("C200", C200, pow2(200) + i(235)),
        ("C200_WRAPPED", C200_WRAPPED, pow2(200) + i(235)),
    ] {
        let c = EcppCertificate::from_pari(text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(c.verify(), "{name} verifies");
        assert_eq!(*c.n(), n, "{name}: n()");
    }
    assert_eq!(
        EcppCertificate::from_pari(C200).unwrap(),
        EcppCertificate::from_pari(C200_WRAPPED).unwrap()
    );
}

/// gp (`t15_certs.gp` of the I1 design, after default(parisizemax, 2^30)): c =
/// primecert(nextprime(2^200)); primecertisvalid of c with c[1][3] doubled, c[1][5][1] + 1,
/// c[1][2] + 2 or c[1][1] + 2 is 0; primecertisvalid(c[2..6]) = 1.
#[test]
fn ecpp_rejects_tampered_certificates() {
    let c = EcppCertificate::from_pari(C200).unwrap();
    let n = c.n().clone();
    let steps = c.steps().to_vec();
    assert_eq!(steps.len(), 6);
    let tampered = |f: &dyn Fn(&mut Vec<rustmath_integers::prime::EcppStep>)| {
        let mut s = steps.clone();
        f(&mut s);
        EcppCertificate::from_steps(s[0].n.clone(), s).verify()
    };
    assert!(!tampered(&|s| s[0].s = s[0].s.clone() * i(2)), "s doubled");
    assert!(
        !tampered(&|s| s[0].x = s[0].x.clone() + Integer::one()),
        "x + 1"
    );
    assert!(!tampered(&|s| s[0].t = s[0].t.clone() + i(2)), "t + 2");
    assert!(!tampered(&|s| s[0].n = s[0].n.clone() + i(2)), "N + 2");
    assert!(!tampered(&|s| s[0].s = Integer::zero()), "s = 0");
    assert!(
        !tampered(&|s| {
            s.remove(2);
        }),
        "step 3 removed"
    );
    // The last step removed: the leaf is step 6's N, which is not below 2^64.
    assert!(
        !tampered(&|s| {
            s.pop();
        }),
        "last step removed"
    );
    // The first step removed but n kept: step 2's N is not n.
    assert!(!EcppCertificate::from_steps(n.clone(), steps[1..].to_vec()).verify());
    // Negative N.
    assert!(!tampered(&|s| s[0].n = -s[0].n.clone()), "negative N");
    // A leaf certificate (no steps) of a composite word: gp factor(1000000016000000063) =
    // [1000000007,1; 1000000009,1].
    assert!(!EcppCertificate::from_steps(z("1000000016000000063"), vec![]).verify());
    assert!(!EcppCertificate::from_steps(z(PSI_12), vec![]).verify());
    // Garbage text.
    for bad in [
        "",
        "[",
        "[]",
        "[[1, 2, 3]]",
        "[[1,2,3,4,[5,6]]] x",
        "12a",
        "[[1,2,3,4,[5,6]],]",
    ] {
        assert!(
            EcppCertificate::from_pari(bad).is_err(),
            "from_pari({bad:?})"
        );
    }
    // Steps 2..6 alone are a valid certificate, of step 1's q, not of N.
    let tail = EcppCertificate::from_steps(steps[1].n.clone(), steps[1..].to_vec());
    assert!(tail.verify());
    assert_ne!(*tail.n(), n);
    let q1 = (steps[0].n.clone() + Integer::one() - steps[0].t.clone()) / steps[0].s.clone();
    assert_eq!(*tail.n(), q1);
}

/// Certificates of the prover: verified, about the input, PARI's layout (PARI accepted the
/// design prototype's certificates of 2^89-1, 2^127-1, the P-256 prime and 2^521-1 with
/// primecertisvalid = 1), every cofactor s >= 2 and the last q below 2^64.
#[test]
fn ecpp_roundtrip_and_pari_shape() {
    for n in [
        pow2(89) - Integer::one(),
        pow2(127) - Integer::one(),
        Integer::from(10).pow(40) + i(121), // gp: nextprime(10^40) - 10^40 = 121
        pow2(200) + i(235),                 // gp: nextprime(2^200) - 2^200 = 235
    ] {
        let Primality::Prime(c) = prove_primality(&n) else {
            panic!("prove_primality({n})");
        };
        assert!(c.verify());
        assert_eq!(*c.n(), n);
        assert_eq!(EcppCertificate::from_pari(&c.to_pari()).unwrap(), c);
        assert!(!c.steps().is_empty());
        for st in c.steps() {
            assert!(st.s >= i(2), "s = {}", st.s);
        }
        let last = c.steps().last().unwrap();
        let q = (last.n.clone() + Integer::one() - last.t.clone()) / last.s.clone();
        assert!(q < pow2(64), "leaf {q}");
    }
}

#[test]
fn ecpp_small_inputs() {
    // gp: isprime(2^61-1) = 1, isprime(2^64-59) = 1
    for p in [i(2), i(3), pow2(61) - Integer::one(), pow2(64) - i(59)] {
        let Primality::Prime(c) = prove_primality(&p) else {
            panic!("prove_primality({p})");
        };
        assert!(c.steps().is_empty());
        assert!(c.verify());
        assert_eq!(c.to_pari(), p.to_string());
        assert_eq!(EcppCertificate::from_pari(&c.to_pari()).unwrap(), c);
    }
}

#[test]
fn is_prime_miller_rabin_counts_bases() {
    // gp: factor(3215031751) = [151,1; 751,1; 28351,1], a strong pseudoprime to 2, 3, 5, 7 and
    // not to 11; factor(3825123056546413051) = [149491,1; 747451,1; 34233211,1], a strong
    // pseudoprime to 2..31 and not to 37.
    let a = i(3_215_031_751);
    assert!(is_prime_miller_rabin(&a, 4));
    assert!(!is_prime_miller_rabin(&a, 5));
    let b = i(3_825_123_056_546_413_051);
    assert!(is_prime_miller_rabin(&b, 11));
    assert!(!is_prime_miller_rabin(&b, 12));
    // gp: isprime(2^127-1) = 1
    assert!(is_prime_miller_rabin(&(pow2(127) - Integer::one()), 20));
    assert!(!is_prime_miller_rabin(&i(1), 20));
    assert!(is_prime_miller_rabin(&i(3), 0));
    assert!(!is_prime_miller_rabin(&i(9), 0));
}

/// MAGMA H18E4: 10-digit safe primes p (p and (p - 1)/2 prime) found with NextPrime and
/// IsPrime. gp (`t17_handbook.gp`): L = List(); p = 10^9; while(#L < 5, p = nextprime(p + 1);
/// if(isprime((p-1)/2), listput(L, p))); L = [1000000007, 1000000403, 1000000787, 1000001447,
/// 1000001819].
#[test]
fn handbook_h18e4_safe_primes() {
    let mut found = Vec::new();
    let mut p = Integer::from(10).pow(9);
    while found.len() < 5 {
        p = next_prime(&p);
        if is_prime(&((p.clone() - Integer::one()) / i(2))) {
            found.push(p.clone());
        }
    }
    let want: Vec<Integer> = [
        1000000007i64,
        1000000403,
        1000000787,
        1000001447,
        1000001819,
    ]
    .into_iter()
    .map(i)
    .collect();
    assert_eq!(found, want);
}

/// MAGMA H18E6: probable repunit primes R_n = (10^n - 1)/9 with IsProbablePrime. gp:
/// select(n -> ispseudoprime((10^n-1)/9), [2..320]) = [2, 19, 23, 317];
/// isprime((10^19-1)/9) = isprime((10^23-1)/9) = 1 (both below psi_12, so a proof).
#[test]
fn handbook_h18e6_probable_repunits() {
    let ten = Integer::from(10);
    let repunit = |n: u32| (ten.pow(n) - Integer::one()) / Integer::from(9);
    let found: Vec<u32> = (2..=320)
        .filter(|&n| is_probable_prime(&repunit(n)))
        .collect();
    assert_eq!(found, vec![2, 19, 23, 317]);
    assert!(is_prime(&repunit(19)));
    assert!(is_prime(&repunit(23)));
}

#[test]
fn next_previous_prime_values() {
    // MAGMA NextPrime(n) is the least prime > n, gp nextprime(n + 1):
    // nextprime(2^64+1) = 18446744073709551629,
    // nextprime(2^127+1) = 170141183460469231731687303715884105757,
    // nextprime(10^40+1) - 10^40 = 121; precprime(2^64-1) = 18446744073709551557,
    // precprime(2^127-1) == 2^127 - 1
    let cases_next = [
        (i(-5), i(2)),
        (i(0), i(2)),
        (i(1), i(2)),
        (i(2), i(3)),
        (pow2(64), z("18446744073709551629")),
        (pow2(127), z("170141183460469231731687303715884105757")),
        (
            Integer::from(10).pow(40),
            Integer::from(10).pow(40) + i(121),
        ),
    ];
    for (n, want) in cases_next {
        assert_eq!(next_prime(&n), want, "next_prime({n})");
        assert_eq!(try_next_prime(&n), Ok(want.clone()), "try_next_prime({n})");
    }
    let cases_prev = [
        (i(3), Some(i(2))),
        (i(2), None),
        (i(-4), None),
        (pow2(64), Some(z("18446744073709551557"))),
        (pow2(64) + i(13), Some(z("18446744073709551557"))),
        (pow2(127), Some(pow2(127) - Integer::one())),
    ];
    for (n, want) in cases_prev {
        assert_eq!(previous_prime(&n), want, "previous_prime({n})");
        assert_eq!(
            try_previous_prime(&n),
            Ok(want.clone()),
            "try_previous_prime({n})"
        );
    }
}

#[test]
fn random_prime_proves_only_the_pick() {
    // gp: primes([10^30, 10^30+299]) - 10^30 = [57, 99, 211, 231, 271]
    let base = Integer::from(10).pow(30);
    let p = try_random_prime(&base, &(base.clone() + i(300)))
        .unwrap()
        .expect("five primes in the window");
    let offset = p.clone() - base.clone();
    assert!(
        [57, 99, 211, 231, 271].map(i).contains(&offset),
        "offset {offset}"
    );
    assert!(is_prime(&p));
    assert_eq!(try_random_prime(&i(24), &i(29)), Ok(None));
    assert_eq!(try_random_prime(&i(20), &i(10)), Ok(None));
    let p = random_prime(&i(10), &i(20)).unwrap();
    assert!([11, 13, 17, 19].map(i).contains(&p));
}

#[test]
fn nth_prime_and_prime_pi_by_sieve() {
    // gp: [prime(1), prime(10), prime(100), prime(10^5), prime(10^6)]
    //     = [2, 29, 541, 1299709, 15485863]; [primepi(10^6), primepi(10^7)] = [78498, 664579]
    for (n, p) in [
        (1usize, 2i64),
        (10, 29),
        (100, 541),
        (100_000, 1_299_709),
        (1_000_000, 15_485_863),
    ] {
        assert_eq!(nth_prime(n), i(p), "nth_prime({n})");
    }
    assert_eq!(prime_pi(&i(1_000_000)), 78498);
    assert_eq!(prime_pi(&i(10_000_000)), 664_579);
}

#[test]
fn prime_range_windows() {
    // gp: primes([10^12, 10^12+99]); primes([2^64-100, 2^64-1]); primes([2^64, 2^64+99])
    let t12 = Integer::from(10).pow(12);
    let cases = [
        (
            t12.clone(),
            t12.clone() + i(100),
            vec![
                "1000000000039",
                "1000000000061",
                "1000000000063",
                "1000000000091",
            ],
        ),
        (
            pow2(64) - i(100),
            pow2(64),
            vec![
                "18446744073709551521",
                "18446744073709551533",
                "18446744073709551557",
            ],
        ),
        (
            pow2(64),
            pow2(64) + i(100),
            vec![
                "18446744073709551629",
                "18446744073709551653",
                "18446744073709551667",
                "18446744073709551697",
                "18446744073709551709",
            ],
        ),
        (
            pow2(64) - i(100),
            pow2(64) + i(40),
            vec![
                "18446744073709551521",
                "18446744073709551533",
                "18446744073709551557",
                "18446744073709551629",
                "18446744073709551653",
            ],
        ),
    ];
    for (lo, hi, want) in cases {
        let want: Vec<Integer> = want.into_iter().map(z).collect();
        assert_eq!(prime_range(&lo, &hi), want, "[{lo}, {hi})");
        assert_eq!(try_prime_range(&lo, &hi), Ok(want.clone()), "[{lo}, {hi})");
    }
}

/// Review 1 B3: stage 1 must use every prime up to the bound, also above 2^16.
/// gp (`pm1.gp` of the I1 design): factor(1835038000112854837) = [917519,1; 2000000000123,1];
/// factor(917519-1) = [2,1; 7,1; 65537,1]; factor(2000000000122) = [2,1; 1000000000061,1];
/// the stage-1 gcd is 917519 for B = 70000 and 1 for B = 65535.
#[test]
fn pollard_p_minus_1_bound_above_2_16() {
    let n = i(1_835_038_000_112_854_837);
    assert_eq!(pollard_p_minus_1(&n, 70000), Some(i(917_519)));
    assert_eq!(pollard_p_minus_1(&n, 65535), None);
    // A prime has no factor to find (gp: isprime(2) = 1).
    assert_eq!(pollard_p_minus_1(&i(2), 100), None);
}

#[test]
fn prime_power_values() {
    // gp: ispower(3^40,,&r) = 40 with r = 3; ispower((2^61-1)^3,,&r) = 3 with r = 2^61-1;
    // isprime(2^61-1) = 1; isprime(2^89-1) = 1; isprime(318665857834031151167461) = 0
    let m61 = pow2(61) - Integer::one();
    assert_eq!(prime_power(&pow2(61)), Some((i(2), 61)));
    assert_eq!(prime_power(&Integer::from(3).pow(40)), Some((i(3), 40)));
    assert_eq!(prime_power(&m61.pow(3)), Some((m61.clone(), 3)));
    assert_eq!(
        prime_power(&(m61.clone() * (pow2(31) - Integer::one()))),
        None
    );
    let m89 = pow2(89) - Integer::one();
    assert_eq!(prime_power(&m89), Some((m89.clone(), 1)));
    assert_eq!(prime_power(&z(PSI_12)), None);
    for n in [1i64, 0, -8] {
        assert_eq!(prime_power(&i(n)), None, "prime_power({n})");
    }
}

/// The cache of proven primes cannot change an answer. A prime no other test in this binary
/// proves (gp: nextprime(10^45) - 10^45 = 9).
#[test]
fn try_is_prime_cache_is_transparent() {
    let p = Integer::from(10).pow(45) + i(9);
    let t0 = std::time::Instant::now();
    assert_eq!(try_is_prime(&p), Ok(true));
    let first = t0.elapsed();
    let t1 = std::time::Instant::now();
    assert_eq!(try_is_prime(&p), Ok(true));
    let second = t1.elapsed();
    println!("try_is_prime(10^45 + 9): first {first:?}, second {second:?}");
}

/// Release timing of one 1024-bit proof (design acceptance: 3 to 5 s). Run with
/// `cargo test --release -p rustmath-integers --test ch18_primality -- --ignored
/// ecpp_1024_release --nocapture`.
#[test]
#[ignore]
fn ecpp_1024_release() {
    // A fixed pseudo-random 1024-bit start: 2^1023 + 3^640 (gp: #binary(3^640) = 1015).
    let mut n = pow2(1023) + Integer::from(3).pow(640);
    if n.is_even() {
        n = n + Integer::one();
    }
    while !is_probable_prime(&n) {
        n = n + i(2);
    }
    let t = std::time::Instant::now();
    let Primality::Prime(c) = prove_primality(&n) else {
        panic!("1024-bit proof unresolved");
    };
    println!(
        "1024-bit prime 2^1023 + 3^640 + {}: {} levels in {:?}",
        n.clone() - pow2(1023) - Integer::from(3).pow(640),
        c.steps().len(),
        t.elapsed()
    );
    assert!(c.verify());
}

/// Prints the prover's certificates for the one-off PARI check of the acceptance list (gp
/// `primecertisvalid`): `cargo test -p rustmath-integers --test ch18_primality -- --ignored
/// print_certificates_for_pari --nocapture`.
#[test]
#[ignore]
fn print_certificates_for_pari() {
    let mut n256 = pow2(255) + Integer::from(7).pow(90); // a fixed 256-bit start
    while !is_probable_prime(&n256) {
        n256 = n256 + Integer::one();
    }
    for n in [
        pow2(127) - Integer::one(),
        Integer::from(10).pow(40) + i(121),
        pow2(200) + i(235),
        n256,
    ] {
        let Primality::Prime(c) = prove_primality(&n) else {
            panic!("{n}");
        };
        println!("CERT {}", c.to_pari());
    }
}
