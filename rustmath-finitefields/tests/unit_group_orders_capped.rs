//! The cost cap of the `Ring::multiplicative_order_status` hooks of the finite rings.
//!
//! MAGMA Handbook ch. 19 §19.7 `Order(a)` and ch. 21 §21.4.7 `Order(a)` (digests
//! `rustmath-lava/docs/Chapter_19_Integer_Residue_Class_Rings.md`, `Chapter_21_Finite_Fields.md`);
//! H. Cohen, GTM 138, Alg. 1.4.3 (order from a factored multiple). The hooks factor the group
//! exponent with a cost cap (the sieve on at most 30 digits, ECPP on at most 256 bits) and
//! answer `Undecided` beyond it; the inherent methods are exact.
//!
//! This file holds exactly one test, on purpose. The factorization cache is process-wide, and
//! each test file is its own process: a second test in this file that factored P43 - 1 exactly
//! could run first and turn phase 1's `Undecided` into a decided answer. A future test that
//! asserts a capped `Undecided` needs a file of its own for the same reason.
//!
//! P43 = 2580000000000000001462000000000000000177763 is prime, and P43 - 1 = 2 · 43 · C41 with
//! C41 = 100000000000000000039 · 300000000000000000053, a 41-digit composite: 8191 rho steps do
//! not split it (its factors have 21 digits) and it is above the 30-digit sieve cap.
//! Reference values, PARI/GP 2.15.4 (`gp -q`):
//! ```text
//! P43=2580000000000000001462000000000000000177763;
//! print([isprime(P43), factor(P43-1)]);
//!     \\ [1, [2,1; 43,1; 100000000000000000039,1; 300000000000000000053,1]]
//! print([znorder(Mod(2,P43))==P43-1, znorder(Mod(3,P43))==(P43-1)/2]);          \\ [1, 1]
//! print([znorder(Mod(1+P43,P43^2))==P43, znorder(Mod(2*P43-1,P43^2))==2*P43,
//!        znorder(Mod(2,P43^2))==P43*(P43-1)]);                                  \\ [1, 1, 1]
//! ```

use rustmath_core::{MultiplicativeOrder, Ring};
use rustmath_finitefields::{FiniteField, GaloisRing, IntegerMod, PrimeField};
use rustmath_integers::Integer;

fn int(n: i64) -> Integer {
    Integer::from(n)
}

#[test]
fn capped_hooks_then_exact_orders_p43() {
    use MultiplicativeOrder::{Finite, FiniteExceedsU64, NotAUnit, Undecided};
    let p = Integer::from_decimal_str("2580000000000000001462000000000000000177763").unwrap();
    let one = Integer::one();
    let pm1 = p.clone() - one.clone();
    let pf = |v: Integer| PrimeField::new(v, p.clone()).unwrap();
    let im = |v: Integer| IntegerMod::new(v, p.clone()).unwrap();
    let ff = FiniteField::new(p.clone(), 1).unwrap();
    let gr = GaloisRing::new(p.clone(), 2, 1).unwrap(); // Z/P43^2

    // Phase 1: no exact call on P43 - 1 yet, so the capped factorization fails.
    for v in [int(2), int(-1), int(1), int(0)] {
        let want = match v.to_string().as_str() {
            "2" => Undecided,
            "-1" => Finite(2),
            "1" => Finite(1),
            _ => NotAUnit,
        };
        assert_eq!(
            pf(v.clone()).multiplicative_order_status(),
            want,
            "PrimeField {v}"
        );
        assert_eq!(
            im(v.clone()).multiplicative_order_status(),
            want,
            "IntegerMod {v}"
        );
        assert_eq!(
            ff.from_int(v.clone()).multiplicative_order_status(),
            want,
            "GF(P43) {v}"
        );
        assert_eq!(
            gr.from_int(v.clone()).multiplicative_order_status(),
            want,
            "Z/P43^2 {v}"
        );
    }
    // Residues 1 and -1 need no factoring: orders P43 and 2·P43.
    let x = gr.from_int(one.clone() + p.clone());
    assert_eq!(x.multiplicative_order_status(), FiniteExceedsU64);
    assert_eq!(x.multiplicative_order(), Some(p.clone()));
    let y = gr.from_int(int(2) * p.clone() - one.clone());
    assert_eq!(y.multiplicative_order_status(), FiniteExceedsU64);
    assert_eq!(y.multiplicative_order(), Some(int(2) * p.clone()));
    assert_eq!(
        gr.from_int(p.clone()).multiplicative_order_status(),
        NotAUnit
    );
    // Still undecided after those (they factored nothing).
    assert_eq!(pf(int(2)).multiplicative_order_status(), Undecided);

    // Phase 2: the exact inherent order factors P43 - 1 (the sieve on C41) and caches it.
    assert_eq!(pf(int(2)).multiplicative_order(), Some(pm1.clone()));
    assert_eq!(
        pf(int(3)).multiplicative_order(),
        Some(pm1.clone() / int(2))
    );
    // The same hooks are now decided: the documented cache effect (both answers honest).
    assert_eq!(pf(int(2)).multiplicative_order_status(), FiniteExceedsU64);
    assert_eq!(im(int(2)).multiplicative_order_status(), FiniteExceedsU64);
    assert_eq!(
        ff.from_int(int(2)).multiplicative_order_status(),
        FiniteExceedsU64
    );
    assert_eq!(
        gr.from_int(int(2)).multiplicative_order_status(),
        FiniteExceedsU64
    );
    assert_eq!(im(int(2)).multiplicative_order(), Some(pm1.clone()));
    assert_eq!(
        ff.from_int(int(2)).multiplicative_order(),
        Some(pm1.clone())
    );
    assert_eq!(
        gr.from_int(int(2)).multiplicative_order(),
        Some(p.clone() * pm1)
    );
}
