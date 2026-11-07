//! Parent finiteness and the generic-path element predicates of the finite rings.
//!
//! MAGMA Handbook: Ch. 17 §17.4.3 `IsFinite`, §17.5.5 `IsUnit`/`IsNilpotent`; Ch. 19 (Z/mZ:
//! `Order`, `IsUnit`); Ch. 21 (finite fields: `IsFinite`, `IsUnit`, `Order`); Ch. 48 (Galois
//! rings: `IsFinite`, `IsUnit`, `IsNilpotent`).
//!
//! These exercise `Parent::is_finite` (three-valued) and the `Ring` hooks `unit_status`,
//! `nilpotent_status`, `unit_inverse`, `multiplicative_order_status`, which generic code
//! reaches (concrete call sites resolve to the inherent methods first). Reference values are
//! from PARI/GP (`echo '<expr>' | gp -q`) or GAP, quoted per test.

use rustmath_core::{MathError, MultiplicativeOrder, Parent, Ring};
use rustmath_finitefields::{
    ExtensionField, FiniteField, FiniteFieldElement, GaloisRing, GaloisRingElement, IntegerMod,
    PrimeField, Zmod,
};
use rustmath_integers::Integer;
use rustmath_matrix::MatrixAlgebra;
use rustmath_polynomials::UnivariatePolynomial;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

fn int(n: i64) -> Integer {
    Integer::from(n)
}

fn two_pow(k: u32) -> Integer {
    Integer::from(2).pow(k)
}

fn imod(v: i64, m: i64) -> IntegerMod {
    IntegerMod::new(int(v), int(m)).unwrap()
}

/// Generic-code view: what `fn f<R: Ring>` sees.
fn hooks<R: Ring>(x: &R) -> (Option<bool>, Option<bool>, MultiplicativeOrder) {
    (x.unit_status(), x.nilpotent_status(), x.multiplicative_order_status())
}

// ---------------------------------------------------------------------------------------
// Defect (a): Parent::is_finite for parents with more than usize::MAX elements
// ---------------------------------------------------------------------------------------

// PARI: isprime(2^64+13) -> 1; 2^64+13 > 2^64-1 -> 1. GAP: IsFinite(GF(2^64+13)) -> true.
#[test]
fn big_prime_field_is_finite() {
    let p = two_pow(64) + int(13);
    let f = FiniteField::new(p.clone(), 1).unwrap();
    assert_eq!(f.cardinality(), None);
    assert_eq!(f.is_finite(), Some(true));
    assert_eq!(f.order(), p);
}

// GAP: Size(GF(2,64)) = 2^64 -> true; PARI: 2^64 > 2^64-1 -> 1.
#[test]
fn gf_2_64_is_finite() {
    let f = FiniteField::new(int(2), 64).unwrap();
    assert_eq!(f.cardinality(), None);
    assert_eq!(f.is_finite(), Some(true));
    assert_eq!(f.order(), two_pow(64));
}

// MAGMA Ch. 48: IsFinite(R) is true for every Galois ring; #GR(2^64,1) = #GR(2,64) = 2^64.
#[test]
fn big_galois_rings_are_finite() {
    let z_2_64 = GaloisRing::new(int(2), 64, 1).unwrap();
    assert_eq!(Parent::cardinality(&z_2_64), None);
    assert_eq!(z_2_64.is_finite(), Some(true));
    assert_eq!(z_2_64.cardinality(), two_pow(64));

    let gf_2_64 = GaloisRing::new(int(2), 1, 64).unwrap();
    assert_eq!(Parent::cardinality(&gf_2_64), None);
    assert_eq!(gf_2_64.is_finite(), Some(true));
}

// MAGMA H48E1, GR(2^3, 2). PARI (gr82.gp): #E -> 64.
#[test]
fn small_galois_ring_cardinality() {
    let r = GaloisRing::new(int(2), 3, 2).unwrap();
    assert_eq!(Parent::cardinality(&r), Some(64));
    assert_eq!(r.is_finite(), Some(true));
}

// PARI: 2^64+1 > 2^64-1 -> 1.
#[test]
fn big_zmod_is_finite() {
    let z = Zmod::new(two_pow(64) + int(1)).unwrap();
    assert_eq!(Parent::cardinality(&z), None);
    assert_eq!(Parent::is_finite(&z), Some(true));
}

// GAP: Size(FullMatrixAlgebra(GF(2),2)) -> 16, Size(FullMatrixAlgebra(GF(3),2)) -> 81. The
// element type PrimeField does not fix p, so the parent cannot say more than "undecided";
// before the fix it claimed `false` (infinite).
#[test]
fn matrix_algebra_over_prime_field_is_undecided_not_infinite() {
    let m = MatrixAlgebra::<PrimeField>::full(2);
    assert_eq!(m.is_finite(), None);
}

// ---------------------------------------------------------------------------------------
// Defect (b): IntegerMod
// ---------------------------------------------------------------------------------------

// PARI: [x | x <- [0..11], gcd(x,12)==1] -> [1, 5, 7, 11]   (GAP: [ 1, 5, 7, 11 ])
//       [x | x <- [0..11], Mod(x,12)^12 == 0] -> [0, 6]
//       [[x, znorder(Mod(x,12))] | x <- [0..11], gcd(x,12)==1] -> [[1,1],[5,2],[7,2],[11,2]]
//       lift(Mod(5,12)^-1) -> 5
//       [x | x <- [0..11], Mod(x,12)^2 == Mod(x,12)] -> [0, 1, 4, 9]
#[test]
fn integer_mod_12_all_residues() {
    for v in 0..12i64 {
        let x = imod(v, 12);
        let unit = [1, 5, 7, 11].contains(&v);
        let nilpotent = [0, 6].contains(&v);
        assert_eq!(x.unit_status(), Some(unit), "unit {v}");
        assert_eq!(x.nilpotent_status(), Some(nilpotent), "nilpotent {v}");
        // the hooks agree with the inherent methods
        assert_eq!(x.unit_status(), Some(x.is_unit()));
        assert_eq!(x.nilpotent_status(), Some(x.is_nilpotent()));
        let expected_order = match v {
            1 => MultiplicativeOrder::Finite(1),
            5 | 7 | 11 => MultiplicativeOrder::Finite(2),
            _ => MultiplicativeOrder::NotAUnit,
        };
        assert_eq!(x.multiplicative_order_status(), expected_order, "order {v}");
        if unit {
            let inv = Ring::unit_inverse(&x).unwrap();
            assert!((x.clone() * inv).is_one());
        } else {
            assert_eq!(Ring::unit_inverse(&x), Err(MathError::NotInvertible));
        }
        let idempotent = x.clone() * x.clone() == x;
        assert_eq!(idempotent, [0, 1, 4, 9].contains(&v), "idempotent {v}");
    }
    assert_eq!(Ring::unit_inverse(&imod(5, 12)).unwrap(), imod(5, 12));
}

// The repro case. PARI: znorder(Mod(3,8)) -> 2; lift(Mod(3,8)^-1) -> 3.
#[test]
fn integer_mod_3_mod_8() {
    let x = imod(3, 8);
    assert_eq!(hooks(&x), (Some(true), Some(false), MultiplicativeOrder::Finite(2)));
    assert_eq!(Ring::unit_inverse(&x).unwrap(), imod(3, 8));
}

// PARI: [x | x <- [0..3], Mod(x,4)^4 == 0] -> [0, 2]   (GAP: [ 0, 2 ])
#[test]
fn integer_mod_2_mod_4_is_nilpotent() {
    let x = imod(2, 4);
    assert_eq!(hooks(&x), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
}

// The modulus-0 sentinel is an integer constant awaiting Z -> Z/nZ; generic code builds 2 as
// R::one() + R::one(). PARI: [gcd(2,5), gcd(2,4), Mod(2,4)^2, Mod(-1,2)==1] -> [1, 2, 0, 1]:
// 2 is a unit mod 5 and nilpotent mod 4, and -1 = 1 mod 2, so none of these is decided.
#[test]
fn integer_mod_sentinel_rules() {
    let one = <IntegerMod as Ring>::one();
    let zero = <IntegerMod as Ring>::zero();
    let two = one.clone() + one.clone();
    let minus_one = -one.clone();

    assert_eq!(hooks(&two), (None, None, MultiplicativeOrder::Undecided));
    assert!(matches!(Ring::unit_inverse(&two), Err(MathError::InvalidArgument(_))));
    // ... and it binds both ways, as PARI says:
    assert!((two.clone() * imod(1, 5)).is_unit());
    assert!((two.clone() * imod(1, 4)).is_nilpotent());

    assert_eq!(hooks(&minus_one), (Some(true), Some(false), MultiplicativeOrder::Undecided));
    assert_eq!(Ring::unit_inverse(&minus_one).unwrap(), minus_one);
    assert!((minus_one.clone() * imod(1, 2)).is_one()); // -1 = 1 in Z/2

    assert_eq!(hooks(&one), (Some(true), Some(false), MultiplicativeOrder::Finite(1)));
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
    assert_eq!(Ring::unit_inverse(&zero), Err(MathError::NotInvertible));
}

// ---------------------------------------------------------------------------------------
// PrimeField (composite moduli are accepted by PrimeField::new)
// ---------------------------------------------------------------------------------------

// PARI: [gcd(2,4), Mod(2,4)^2 == 0] -> [2, 1]; znorder(Mod(3,7)) -> 6; lift(Mod(3,7)^-1) -> 5.
#[test]
fn prime_field_predicates() {
    let x = PrimeField::new(int(2), int(4)).unwrap();
    assert_eq!(hooks(&x), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
    assert_eq!(Ring::unit_inverse(&x), Err(MathError::NotInvertible));

    let y = PrimeField::new(int(3), int(7)).unwrap();
    assert_eq!(hooks(&y), (Some(true), Some(false), MultiplicativeOrder::Finite(6)));
    assert_eq!(Ring::unit_inverse(&y).unwrap(), PrimeField::new(int(5), int(7)).unwrap());
    let z = PrimeField::new(int(0), int(7)).unwrap();
    assert_eq!(hooks(&z), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
}

// Same sentinel rules as IntegerMod (PARI line in `integer_mod_sentinel_rules`).
#[test]
fn prime_field_sentinel_rules() {
    let one = <PrimeField as Ring>::one();
    let two = one.clone() + one.clone();
    assert_eq!(hooks(&two), (None, None, MultiplicativeOrder::Undecided));
    assert!(matches!(Ring::unit_inverse(&two), Err(MathError::InvalidArgument(_))));
    assert_eq!(hooks(&-one.clone()), (Some(true), Some(false), MultiplicativeOrder::Undecided));
    assert_eq!(hooks(&one), (Some(true), Some(false), MultiplicativeOrder::Finite(1)));
    let zero = <PrimeField as Ring>::zero();
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
}

// ---------------------------------------------------------------------------------------
// FiniteFieldElement
// ---------------------------------------------------------------------------------------

// PARI: [znorder(Mod(2,5)), lift(Mod(2,5)^-1)] -> [4, 3]; [x | x <- [0..4], Mod(x,5)^5==0] -> [0].
#[test]
fn gf5_predicates() {
    let f = FiniteField::new(int(5), 1).unwrap();
    let two = f.from_int(int(2));
    assert_eq!(hooks(&two), (Some(true), Some(false), MultiplicativeOrder::Finite(4)));
    assert_eq!(Ring::unit_inverse(&two).unwrap(), f.from_int(int(3)));
    let zero = f.zero();
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
    assert_eq!(Ring::unit_inverse(&zero), Err(MathError::NotInvertible));
    for v in 1..5 {
        assert_eq!(f.from_int(int(v)).nilpotent_status(), Some(false));
    }
}

// Regression for the deleted search bound of 1000 in rustmath-category.
// PARI: znorder(Mod(3,1013)) -> 1012; Mod(3,1013)^1000 == 1 -> 0.
#[test]
fn gf1013_order_beyond_1000() {
    let f = FiniteField::new(int(1013), 1).unwrap();
    assert_eq!(f.from_int(int(3)).multiplicative_order_status(), MultiplicativeOrder::Finite(1012));
}

// p = 12*2^64 + 1, p - 1 = 2^66 * 3 (so the trial-division factoriser is instant).
// PARI: isprime(p) -> 1; znorder(Mod(2,p)) -> 110680464442257309696 (> 2^64 - 1);
//       t = lift(Mod(10,p)^(2^64)) -> 128622681397817100675; znorder(Mod(t,p)) -> 12.
#[test]
fn order_beyond_u64() {
    let p = int(12) * two_pow(64) + int(1);
    let f = FiniteField::new(p, 1).unwrap();
    let two = f.from_int(int(2));
    assert_eq!(two.multiplicative_order_status(), MultiplicativeOrder::FiniteExceedsU64);
    // 110680464442257309696 = 6 * 2^64
    assert_eq!(two.multiplicative_order(), Some(int(6) * two_pow(64)));
    // 128622681397817100675 = 6 * 2^64 + 17942216955559790979
    let t = f.from_int(int(6) * two_pow(64) + Integer::from(17942216955559790979u64));
    assert_eq!(t.multiplicative_order_status(), MultiplicativeOrder::Finite(12));
}

// An unbound constant c awaits Z -> GF(p^n): -1 has order 1 in characteristic 2, and
// 2 = one + one is 0 in GF(2^n) but a unit in GF(3), so only the axioms are decided.
#[test]
fn unbound_finite_field_constants() {
    let one = <FiniteFieldElement as Ring>::one();
    assert_eq!(hooks(&one), (Some(true), Some(false), MultiplicativeOrder::Finite(1)));
    let minus_one = -one.clone();
    assert_eq!(hooks(&minus_one), (Some(true), Some(false), MultiplicativeOrder::Undecided));
    assert_eq!(Ring::unit_inverse(&minus_one).unwrap(), minus_one);
    let zero = <FiniteFieldElement as Ring>::zero();
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
    let two = one.clone() + one.clone();
    assert_eq!(hooks(&two), (None, None, MultiplicativeOrder::Undecided));
    assert!(matches!(Ring::unit_inverse(&two), Err(MathError::InvalidArgument(_))));
    let gf2 = FiniteField::new(int(2), 1).unwrap();
    assert!((two.clone() * gf2.one()).is_zero());
}

// ExtensionField delegates to FiniteFieldElement. GF(4) = GF(2)[x]/(x^2+x+1); x has order 3
// (GAP: Order(Z(4)) -> 3).
#[test]
fn extension_field_delegates() {
    let irr = UnivariatePolynomial::new(vec![int(1), int(1), int(1)]);
    let x = ExtensionField::new(UnivariatePolynomial::new(vec![int(0), int(1)]), int(2), irr.clone())
        .unwrap();
    assert_eq!(hooks(&x), (Some(true), Some(false), MultiplicativeOrder::Finite(3)));
    let inv = Ring::unit_inverse(&x).unwrap();
    assert!((x.clone() * inv).is_one());
    let zero = ExtensionField::new(UnivariatePolynomial::new(vec![int(0)]), int(2), irr).unwrap();
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
}

// ---------------------------------------------------------------------------------------
// GaloisRingElement
// ---------------------------------------------------------------------------------------

/// All 64 elements of GR(8,2), indexed as PARI's gr82.gp: k - 1 = a0 + 8*a1 <-> a0 + a1*w.
fn all_gr_8_2(r: &GaloisRing) -> Vec<GaloisRingElement> {
    let mut v = Vec::new();
    for a1 in 0..8 {
        for a0 in 0..8 {
            v.push(r.element(vec![int(a0), int(a1)]));
        }
    }
    v
}

/// Brute-force order (complete by Lagrange: it divides #R^* = `bound`); 0 for a non-unit.
fn brute_order(x: &GaloisRingElement, bound: u64) -> u64 {
    let mut y = x.clone();
    for n in 1..=bound {
        if y.is_one() {
            return n;
        }
        y = y * x.clone();
    }
    0
}

// MAGMA H48E1, GR(2^3, 2). PARI (gr82.gp): [#E, units, nilpotents, idempotents] ->
// [64, 48, 16, 2]; orders, k = 1..64:
const GR_8_2_ORDERS: [u64; 64] = [
    0, 1, 0, 2, 0, 2, 0, 2, 3, 6, 12, 12, 6, 6, 12, 12, 0, 4, 0, 4, 0, 4, 0, 4, 6, 12, 12, 6, 6,
    12, 12, 6, 0, 2, 0, 2, 0, 2, 0, 2, 6, 6, 12, 12, 6, 6, 12, 12, 0, 4, 0, 4, 0, 4, 0, 4, 6, 12,
    12, 6, 6, 12, 12, 3,
];

#[test]
fn gr_8_2_exhaustive() {
    let r = GaloisRing::new(int(2), 3, 2).unwrap();
    let elems = all_gr_8_2(&r);
    let (mut units, mut nilpotents, mut idempotents) = (0, 0, 0);
    for (k, x) in elems.iter().enumerate() {
        let u = x.unit_status().unwrap();
        let n = x.nilpotent_status().unwrap();
        assert_eq!(u, x.is_unit());
        assert_eq!(n, x.is_nilpotent());
        units += u as u32;
        nilpotents += n as u32;
        idempotents += x.is_idempotent() as u32;
        let expected = match GR_8_2_ORDERS[k] {
            0 => MultiplicativeOrder::NotAUnit,
            o => MultiplicativeOrder::Finite(o),
        };
        assert_eq!(x.multiplicative_order_status(), expected, "element {k}: {x}");
        assert_eq!(brute_order(x, 48), GR_8_2_ORDERS[k]);
        if u {
            let inv = Ring::unit_inverse(x).unwrap();
            assert!((x.clone() * inv).is_one());
        } else {
            assert_eq!(Ring::unit_inverse(x), Err(MathError::NotInvertible));
        }
    }
    assert_eq!((units, nilpotents, idempotents), (48, 16, 2));
}

// PARI (gr82.gp): [ordr(w), ordr(1+2*w), ordr(w*(1+2*w)), ordr(1+4*w)] -> [3, 4, 12, 2];
// znorder(Mod(3,8)) -> 2. GR(4,2): w -> 3, 1+2w -> 2, w(1+2w) -> 6 (PARI:
// f4 = Mod(1,4)*(x^2+x+1); w4 = Mod(Mod(1,4)*x, f4); [o4(w4), o4(1+2*w4), o4(w4*(1+2*w4))]
// -> [3, 2, 6]).
#[test]
fn gr_orders_named_elements() {
    let r = GaloisRing::new(int(2), 3, 2).unwrap();
    let w = r.generator();
    let one = r.one();
    let two = r.from_int(int(2));
    let four = r.from_int(int(4));
    let ord = |x: GaloisRingElement| x.multiplicative_order_status();
    assert_eq!(ord(w.clone()), MultiplicativeOrder::Finite(3));
    assert_eq!(ord(one.clone() + two.clone() * w.clone()), MultiplicativeOrder::Finite(4));
    assert_eq!(
        ord(w.clone() * (one.clone() + two.clone() * w.clone())),
        MultiplicativeOrder::Finite(12)
    );
    assert_eq!(ord(one.clone() + four * w.clone()), MultiplicativeOrder::Finite(2));
    assert_eq!(ord(r.from_int(int(3))), MultiplicativeOrder::Finite(2));

    let r4 = GaloisRing::new(int(2), 2, 2).unwrap();
    let w4 = r4.generator();
    let u = r4.one() + r4.from_int(int(2)) * w4.clone();
    assert_eq!(ord(w4.clone()), MultiplicativeOrder::Finite(3));
    assert_eq!(ord(u.clone()), MultiplicativeOrder::Finite(2));
    assert_eq!(ord(w4 * u), MultiplicativeOrder::Finite(6));
}

// Odd p: GR(3^3, 2) = (Z/27)[x]/(x^2+2x+2), #R^* = 8 * 81 = 648. PARI histogram of orders
// (0 = non-unit) over the 729 elements:
//   [[0, 81], [1, 1], [2, 1], [3, 8], [4, 2], [6, 8], [8, 4], [9, 72], [12, 16], [18, 72],
//    [24, 32], [36, 144], [72, 288]]
// and [ord(w), ord(1+3w), ord(2), ord(w*(1+3w))] -> [72, 9, 18, 72]
// (scratch script: g = Mod(1,27)*(x^2+2*x+2); ord27(z) = for(n=1,648, if(z^n==1, return(n))); 0).
#[test]
fn gr_27_2_orders_match_pari_and_brute_force() {
    let r = GaloisRing::new(int(3), 3, 2).unwrap();
    // The default modulus is the Conway polynomial of GF(9), x^2 + 2x + 2, as in PARI's g.
    let w = r.generator();
    assert!((w.clone() * w.clone() + r.from_int(int(2)) * w.clone() + r.from_int(int(2))).is_zero());
    let mut hist = std::collections::BTreeMap::<u64, u32>::new();
    for a1 in 0..27 {
        for a0 in 0..27 {
            let x = r.element(vec![int(a0), int(a1)]);
            let o = match x.multiplicative_order_status() {
                MultiplicativeOrder::NotAUnit => 0,
                MultiplicativeOrder::Finite(o) => o,
                other => panic!("undecided order {other:?} for {x}"),
            };
            assert_eq!(o, brute_order(&x, 648), "{x}");
            *hist.entry(o).or_default() += 1;
        }
    }
    let expected: Vec<(u64, u32)> = vec![
        (0, 81), (1, 1), (2, 1), (3, 8), (4, 2), (6, 8), (8, 4), (9, 72), (12, 16), (18, 72),
        (24, 32), (36, 144), (72, 288),
    ];
    assert_eq!(hist.into_iter().collect::<Vec<_>>(), expected);
    let u = r.one() + r.from_int(int(3)) * w.clone();
    assert_eq!(w.multiplicative_order_status(), MultiplicativeOrder::Finite(72));
    assert_eq!(u.multiplicative_order_status(), MultiplicativeOrder::Finite(9));
    assert_eq!(r.from_int(int(2)).multiplicative_order_status(), MultiplicativeOrder::Finite(18));
    assert_eq!((w * u).multiplicative_order_status(), MultiplicativeOrder::Finite(72));
}

// GR(2^64, 1) = Z/2^64: 62 p-power steps. PARI: znorder(Mod(3,2^64)) -> 4611686018427387904
// = 2^62.
#[test]
fn gr_2_64_order_of_3() {
    let r = GaloisRing::new(int(2), 64, 1).unwrap();
    assert_eq!(
        r.from_int(int(3)).multiplicative_order_status(),
        MultiplicativeOrder::Finite(1u64 << 62)
    );
    assert_eq!(r.from_int(int(3)).multiplicative_order(), Some(two_pow(62)));
}

// Orders of at least 2^64: the hook says `FiniteExceedsU64` and the inherent
// `GaloisRingElement::multiplicative_order` gives the exact value.
// PARI: [znorder(Mod(3,2^66)), znorder(Mod(5,2^66)), znorder(Mod(7,2^66)), znorder(Mod(-1,2^66))]
//   -> [18446744073709551616, 18446744073709551616, 9223372036854775808, 2]
//   = [2^64, 2^64, 2^63, 2].
#[test]
fn gr_2_66_orders_beyond_u64() {
    let r = GaloisRing::new(int(2), 66, 1).unwrap();
    let three = r.from_int(int(3));
    assert_eq!(three.multiplicative_order_status(), MultiplicativeOrder::FiniteExceedsU64);
    assert_eq!(three.multiplicative_order(), Some(two_pow(64)));
    let five = r.from_int(int(5));
    assert_eq!(five.multiplicative_order_status(), MultiplicativeOrder::FiniteExceedsU64);
    assert_eq!(five.multiplicative_order(), Some(two_pow(64)));
    // 2^63 still fits in u64.
    let seven = r.from_int(int(7));
    assert_eq!(seven.multiplicative_order_status(), MultiplicativeOrder::Finite(1u64 << 63));
    assert_eq!(seven.multiplicative_order(), Some(two_pow(63)));
    assert_eq!(r.from_int(int(-1)).multiplicative_order(), Some(int(2)));
    assert_eq!(r.one().multiplicative_order(), Some(int(1)));
    // Non-units have no order (MAGMA `Order` returns 0).
    assert_eq!(r.from_int(int(2)).multiplicative_order(), None);
    assert_eq!(r.zero().multiplicative_order(), None);
}

// GR(3^42, 2) = (Z/3^42)[x]/(x^2+2x+2), w the class of x. PARI:
//   g = Mod(1,3^42)*(x^2+2*x+2); w = Mod(Mod(1,3^42)*x, g); y = 1+3*w; z = w*y;
//   [y^(3^41)==1, y^(3^40)==1] -> [1, 0], so ord(1+3w) = 3^41 = 36472996377170786403;
//   [z^(8*3^41)==1, z^(4*3^41)==1, z^(8*3^40)==1] -> [1, 0, 0], so
//   ord(w(1+3w)) = 8*3^41 = 291783971017366291224.
#[test]
fn gr_3_42_2_orders_beyond_u64() {
    let r = GaloisRing::new(int(3), 42, 2).unwrap();
    let w = r.generator();
    assert!((w.clone() * w.clone() + r.from_int(int(2)) * w.clone() + r.from_int(int(2))).is_zero());
    let three_pow_41 = int(3).pow(41);
    assert_eq!(three_pow_41, Integer::from_decimal_str("36472996377170786403").unwrap());
    let y = r.one() + r.from_int(int(3)) * w.clone();
    assert_eq!(y.multiplicative_order_status(), MultiplicativeOrder::FiniteExceedsU64);
    assert_eq!(y.multiplicative_order(), Some(three_pow_41.clone()));
    let z = w * y;
    assert_eq!(z.multiplicative_order_status(), MultiplicativeOrder::FiniteExceedsU64);
    let ord_z = int(8) * three_pow_41;
    assert_eq!(ord_z, Integer::from_decimal_str("291783971017366291224").unwrap());
    assert_eq!(z.multiplicative_order(), Some(ord_z));
}

// The inherent order and the `Ring` hook agree on every element of GR(3^3, 2) (orders
// cross-checked against PARI in `gr_27_2_orders_match_pari_and_brute_force`).
#[test]
fn gr_27_2_inherent_order_matches_hook() {
    let r = GaloisRing::new(int(3), 3, 2).unwrap();
    for a1 in 0..27 {
        for a0 in 0..27 {
            let x = r.element(vec![int(a0), int(a1)]);
            let expected = match x.multiplicative_order_status() {
                MultiplicativeOrder::NotAUnit => None,
                MultiplicativeOrder::Finite(o) => Some(Integer::from(o)),
                other => panic!("undecided order {other:?} for {x}"),
            };
            assert_eq!(x.multiplicative_order(), expected, "{x}");
        }
    }
}

/// GR(8,2) elements seen through a `Ring` impl that overrides none of the hooks and whose
/// static constructors panic (as `GaloisRingElement`'s do): the trait defaults must neither
/// panic nor decide anything beyond the ring axioms.
#[derive(Clone, Debug, PartialEq)]
struct GrDefaults(GaloisRingElement);

impl fmt::Display for GrDefaults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for GrDefaults {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        GrDefaults(self.0 + o.0)
    }
}
impl Sub for GrDefaults {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        GrDefaults(self.0 - o.0)
    }
}
impl Mul for GrDefaults {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        GrDefaults(self.0 * o.0)
    }
}
impl Neg for GrDefaults {
    type Output = Self;
    fn neg(self) -> Self {
        GrDefaults(-self.0)
    }
}
impl Ring for GrDefaults {
    fn zero() -> Self {
        panic!("GrDefaults::zero() needs a parent")
    }
    fn one() -> Self {
        panic!("GrDefaults::one() needs a parent")
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}

#[test]
fn trait_defaults_on_galois_ring_do_not_panic() {
    let r = GaloisRing::new(int(2), 3, 2).unwrap();
    let w = GrDefaults(r.generator());
    let two = GrDefaults(r.from_int(int(2)));
    let minus_one = GrDefaults(-r.one());
    // w is a unit of order 3 and 2 a nilpotent non-unit, but the defaults cannot know.
    assert_eq!(hooks(&w), (None, None, MultiplicativeOrder::Undecided));
    assert_eq!(hooks(&two), (None, None, MultiplicativeOrder::Undecided));
    assert!(matches!(w.unit_inverse(), Err(MathError::NotImplemented(_))));
    // -1 is a unit and not nilpotent; its order (2 here) needs the characteristic.
    assert_eq!(hooks(&minus_one), (Some(true), Some(false), MultiplicativeOrder::Undecided));
    assert_eq!(minus_one.unit_inverse().unwrap(), minus_one);
    let one = GrDefaults(r.one());
    assert_eq!(hooks(&one), (Some(true), Some(false), MultiplicativeOrder::Finite(1)));
    let zero = GrDefaults(r.zero());
    assert_eq!(hooks(&zero), (Some(false), Some(true), MultiplicativeOrder::NotAUnit));
    assert_eq!(zero.unit_inverse(), Err(MathError::NotInvertible));
}
