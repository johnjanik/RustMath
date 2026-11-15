//! Multiplicative orders from the factored group exponent.
//!
//! MAGMA Handbook ch. 19 §19.7 `Order(a)` (the order of a unit of Z/mZ, 0 for a non-unit) and
//! `IsPrimitive(n)`; ch. 21 §21.4.4 `IsPrimitive(a)` and §21.4.7 `Order(a)`, `FactoredOrder(a)`
//! (digests `rustmath-lava/docs/Chapter_19_Integer_Residue_Class_Rings.md`,
//! `Chapter_21_Finite_Fields.md`). Algorithm: H. Cohen, *A Course in Computational Algebraic
//! Number Theory*, GTM 138, Alg. 1.4.3, on `m - 1` (after a Fermat test), the Carmichael
//! function λ(m), or `q - 1 = ∏_{d | n} Φ_d(p)`.
//!
//! Nothing in this file asserts an `Undecided`, so the process-wide factorization cache cannot
//! change an expected value (the capped answers are tested in `unit_group_orders_capped.rs`).
//!
//! Reference values: PARI/GP 2.15.4 (`gp -q`), the script
//! ```text
//! print([znorder(Mod(2,1000003)), znorder(Mod(2,10^9+7)), znorder(Mod(5,10^9+7)),
//!        znorder(Mod(3,2^66))==2^64]);                  \\ [1000002, 500000003, 1000000006, 1]
//! print([znorder(Mod(2,3^40*(10^9+7))), znorder(Mod(5,2^20*3^5*1000003))]);
//!                                       \\ [4052555177334307185113857602, 3538951077888]
//! print([znorder(Mod(2,15)), znorder(Mod(7,15)), znorder(Mod(2,341)), znorder(Mod(3,341)),
//!        znorder(Mod(2,561)), znorder(Mod(5,561))]);     \\ [4, 4, 10, 30, 40, 80]
//! print([znorder(Mod(3,2000000000123)), znorder(Mod(4,20000000000002643))]);
//!                                                       \\ [1000000000061, 10000000000001321]
//! p30=2000000000000000000000000000543; print(znorder(Mod(2,p30)));
//!                                                       \\ 1000000000000000000000000000271
//! w=ffgen(Mod(1,5)*(t^30+t^3+t+3),'t); print([fforder(w), fforder(w+1)]);
//!                                       \\ [133046082087925502232, 14782898009769500248]
//! print(factor(fforder(w+1)));  \\ [2,3; 11,1; 31,1; 61,1; 71,1; 181,1; 521,1; 1741,1; 7621,1]
//! w=ffgen(Mod(1,3)*(t^40+t+2),'t); print(fforder(w));   \\ 12157665459056928800
//! w=ffgen(Mod(1,2)*(t^127+t+1),'t); print(fforder(w)==2^127-1);   \\ 1
//! w=ffgen(Mod(1,101)*(t^12+t+11),'t); print(fforder(w));  \\ 1126825030131969720661200
//! w=ffgen(Mod(1,5)*(t^3+3*t+3),'t); print([fforder(w), fforder(w+1), factor(fforder(w+1))]);
//!                                                       \\ [124, 31, Mat([31, 1])]
//! P43=2580000000000000001462000000000000000177763;
//! print([isprime(P43), znorder(Mod(2,P43))==P43-1, znorder(Mod(3,P43))==(P43-1)/2]); \\ [1,1,1]
//! ```
//! Every defining polynomial above is irreducible (gp `polisirreducible` = 1).

use rustmath_core::Ring;
use rustmath_finitefields::{FiniteField, GaloisRing, IntegerMod, PrimeField};
use rustmath_integers::Integer;

fn int(n: i64) -> Integer {
    Integer::from(n)
}

fn big(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap()
}

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&c| int(c)).collect()
}

const P43: &str = "2580000000000000001462000000000000000177763";

#[test]
fn integer_mod_orders_match_pari() {
    let rows: Vec<(Integer, Integer, Integer)> = vec![
        (int(2), int(1000003), int(1000002)),
        (int(2), int(1000000007), int(500000003)),
        (int(5), int(1000000007), int(1000000006)),
        (int(3), int(2).pow(66), int(2).pow(64)),
        (
            int(2),
            int(3).pow(40) * int(1000000007),
            big("4052555177334307185113857602"),
        ),
        (
            int(5),
            int(2).pow(20) * int(3).pow(5) * int(1000003),
            int(3538951077888),
        ),
        (int(2), int(15), int(4)),
        (int(7), int(15), int(4)),
        // Fermat path: 2^340 = 1 mod 341 = 11 * 31, so h = 340 (gp Mod(2,341)^340 = 1).
        (int(2), int(341), int(10)),
        // λ path: 3^340 = 56 mod 341 (gp), so h = λ(341) = 30.
        (int(3), int(341), int(30)),
        // 561 = 3 * 11 * 17 is a Carmichael number: every unit takes the Fermat path.
        (int(2), int(561), int(40)),
        (int(5), int(561), int(80)),
    ];
    for (a, m, want) in rows {
        let x = IntegerMod::new(a.clone(), m.clone()).unwrap();
        assert_eq!(x.multiplicative_order(), Some(want.clone()), "{a} mod {m}");
    }
    // A non-unit has no order (MAGMA: 0).
    assert_eq!(
        IntegerMod::new(int(6), int(15))
            .unwrap()
            .multiplicative_order(),
        None
    );
}

#[test]
fn prime_field_orders_match_pari() {
    let rows: Vec<(i64, i64, i64)> = vec![
        (3, 2000000000123, 1000000000061),
        (4, 20000000000002643, 10000000000001321),
        // PrimeField::new does not check primality; the orders are still exact.
        (2, 15, 4),
        (2, 341, 10),
        (3, 341, 30),
    ];
    for (a, m, want) in rows {
        let x = PrimeField::new(int(a), int(m)).unwrap();
        assert_eq!(x.multiplicative_order(), Some(int(want)), "{a} mod {m}");
    }
    assert_eq!(
        PrimeField::new(int(5), int(15))
            .unwrap()
            .multiplicative_order(),
        None
    );
    assert_eq!(
        PrimeField::new(int(0), int(7))
            .unwrap()
            .multiplicative_order(),
        None
    );
}

#[test]
fn finite_field_orders_match_pari() {
    // GF(p16), p16 - 1 = 2 * q with q prime (HEAD: 10^8 trial divisions, 21.8 s).
    let f16 = FiniteField::new(int(20000000000002643), 1).unwrap();
    assert_eq!(
        f16.from_int(int(4)).multiplicative_order(),
        Some(int(10000000000001321))
    );
    // GF(p30), p30 = 2q + 1 (HEAD: about 1.4 * 10^15 trial divisions).
    let f30 = FiniteField::new(big("2000000000000000000000000000543"), 1).unwrap();
    assert_eq!(
        f30.from_int(int(2)).multiplicative_order(),
        Some(big("1000000000000000000000000000271"))
    );
    // The default (non-Conway) moduli, asserted first.
    let f5 = FiniteField::new(int(5), 30).unwrap();
    let mut m5 = vec![int(0); 31];
    m5[0] = int(3);
    m5[1] = int(1);
    m5[3] = int(1);
    m5[30] = int(1);
    assert_eq!(f5.defining_polynomial(), &m5[..]);
    let w = f5.generator();
    assert_eq!(w.multiplicative_order(), Some(big("133046082087925502232")));
    assert_eq!(
        (w.clone() + f5.one()).multiplicative_order(),
        Some(big("14782898009769500248"))
    );

    let f3 = FiniteField::new(int(3), 40).unwrap();
    let mut m3 = vec![int(0); 41];
    m3[0] = int(2);
    m3[1] = int(1);
    m3[40] = int(1);
    assert_eq!(f3.defining_polynomial(), &m3[..]);
    assert_eq!(
        f3.generator().multiplicative_order(),
        Some(big("12157665459056928800"))
    );

    let f2 = FiniteField::new(int(2), 127).unwrap();
    let mut m2 = vec![int(0); 128];
    m2[0] = int(1);
    m2[1] = int(1);
    m2[127] = int(1);
    assert_eq!(f2.defining_polynomial(), &m2[..]);
    assert_eq!(
        f2.generator().multiplicative_order(),
        Some(int(2).pow(127) - Integer::one())
    );

    let f101 = FiniteField::new(int(101), 12).unwrap();
    let mut m101 = vec![int(0); 13];
    m101[0] = int(11);
    m101[1] = int(1);
    m101[12] = int(1);
    assert_eq!(f101.defining_polynomial(), &m101[..]);
    assert_eq!(
        f101.generator().multiplicative_order(),
        Some(big("1126825030131969720661200"))
    );
}

#[test]
fn factored_order_matches_pari() {
    let f5 = FiniteField::new(int(5), 30).unwrap();
    let w1 = f5.generator() + f5.one();
    let want: Vec<(Integer, u32)> = [2i64, 11, 31, 61, 71, 181, 521, 1741, 7621]
        .iter()
        .map(|&p| (int(p), if p == 2 { 3 } else { 1 }))
        .collect();
    assert_eq!(w1.factored_multiplicative_order(), Some(want));

    // GF(125) with the Conway polynomial t^3 + 3t + 3.
    let f125 = FiniteField::new(int(5), 3).unwrap();
    assert_eq!(f125.defining_polynomial(), &ints(&[3, 3, 0, 1])[..]);
    let v1 = f125.generator() + f125.one();
    assert_eq!(v1.factored_multiplicative_order(), Some(vec![(int(31), 1)]));
    // MAGMA FactoredOrder(1) is the empty sequence (gp factor(1) = matrix(0,2)).
    assert_eq!(f125.one().factored_multiplicative_order(), Some(vec![]));
    assert_eq!(f125.zero().factored_multiplicative_order(), None);
    // -1 has order 2 without factoring.
    assert_eq!(
        (-f125.one()).factored_multiplicative_order(),
        Some(vec![(int(2), 1)])
    );
}

#[test]
fn is_primitive_and_is_generator() {
    let f125 = FiniteField::new(int(5), 3).unwrap();
    assert!(f125.generator().is_primitive());
    assert!(!(f125.generator() + f125.one()).is_primitive());
    assert!(!f125.zero().is_primitive());
    // In GF(2), 1 generates the trivial group GF(2)^*.
    let f2 = FiniteField::new(int(2), 1).unwrap();
    assert!(f2.one().is_primitive());

    // gp: znorder(Mod(2,P43)) == P43-1, znorder(Mod(3,P43)) == (P43-1)/2. This runs the sieve
    // once on the 41-digit cofactor of P43 - 1 (well below a second in debug).
    let p43 = big(P43);
    assert!(PrimeField::new(int(2), p43.clone()).unwrap().is_generator());
    assert!(!PrimeField::new(int(3), p43.clone()).unwrap().is_generator());
    // A composite modulus has no generator of order m - 1.
    assert!(!PrimeField::new(int(2), int(15)).unwrap().is_generator());
    assert!(PrimeField::new(int(3), int(7)).unwrap().is_generator());
}

/// Brute force: the least k >= 1 with x^k = 1.
fn brute<T: Clone + std::ops::Mul<Output = T>>(x: &T, is_one: impl Fn(&T) -> bool) -> Integer {
    let mut y = x.clone();
    let mut k = 1i64;
    while !is_one(&y) {
        y = y * x.clone();
        k += 1;
    }
    int(k)
}

fn order_from_integer(k: &Integer) -> rustmath_core::MultiplicativeOrder {
    match k.to_string().parse::<u64>() {
        Ok(v) => rustmath_core::MultiplicativeOrder::Finite(v),
        Err(_) => rustmath_core::MultiplicativeOrder::FiniteExceedsU64,
    }
}

/// The capped hooks agree with the exact inherent orders, and both with brute force, on
/// every unit of Z/mZ for m < 200 (Fermat liars such as 3 mod 91 included), every nonzero
/// element of GF(3^4), and every unit of GR(27, 2).
#[test]
fn hooks_agree_with_inherent_small() {
    // gp: Mod(3,91)^90 = 1 and 91 = 7 * 13 (3 is a Fermat liar mod 91).
    assert!(IntegerMod::new(int(3), int(91))
        .unwrap()
        .pow(&int(90))
        .unwrap()
        .value()
        .is_one());
    for m in 2..200i64 {
        for a in 1..m {
            let x = IntegerMod::new(int(a), int(m)).unwrap();
            if !x.is_unit() {
                assert_eq!(x.multiplicative_order(), None);
                continue;
            }
            let inherent = x.multiplicative_order().unwrap();
            assert_eq!(inherent, brute(&x, |y| y.value().is_one()), "{a} mod {m}");
            assert_eq!(
                x.multiplicative_order_status(),
                order_from_integer(&inherent)
            );
            let y = PrimeField::new(int(a), int(m)).unwrap();
            assert_eq!(
                y.multiplicative_order(),
                Some(inherent.clone()),
                "{a} mod {m}"
            );
            assert_eq!(
                y.multiplicative_order_status(),
                order_from_integer(&inherent)
            );
        }
    }
    let f81 = FiniteField::new(int(3), 4).unwrap();
    for i in 1..81i64 {
        let x = f81.element(ints(&[i % 3, (i / 3) % 3, (i / 9) % 3, i / 27]));
        let inherent = x.multiplicative_order().unwrap();
        assert_eq!(inherent, brute(&x, |y| y.is_one()));
        assert_eq!(
            x.multiplicative_order_status(),
            order_from_integer(&inherent)
        );
    }
    let gr = GaloisRing::new(int(3), 3, 2).unwrap();
    let mut units = 0;
    for a in 0..27i64 {
        for b in 0..27i64 {
            let x = gr.element(ints(&[a, b]));
            match x.multiplicative_order() {
                None => assert!(!x.is_unit()),
                Some(inherent) => {
                    units += 1;
                    assert_eq!(inherent, brute(&x, |y| y.is_one()));
                    assert_eq!(
                        x.multiplicative_order_status(),
                        order_from_integer(&inherent)
                    );
                }
            }
        }
    }
    // |GR(27, 2)^*| = 27^2 - 9^2 = 648
    assert_eq!(units, 648);
}
