//! `GaloisRingElement`'s static `Ring::zero()`/`Ring::one()`: unbound integer constants.
//!
//! MAGMA Handbook ch. 48 §48.2.3 (`Chapter_48_Galois_Rings.md`): `Zero(R)`, `One(R)`, and
//! `R ! a` for an integer `a` ("identified with a mod p^a"). The static constructors cannot
//! know `p`, `a`, `d`, so (as `FiniteFieldElement` and the `IntegerMod`/`PrimeField` modulus-0
//! sentinel since 5ff8749) they return an integer constant `c` awaiting the canonical map
//! `Z -> GR(p^a, d)`, `c ↦ c·1`, which is the unique ring homomorphism from `Z`. Before this
//! change they panicked, so `fold(R::zero(), +)`, `Ring::pow(x, 0)` and
//! `UnivariatePolynomial<GaloisRingElement>` multiplication aborted.
//!
//! # Reference values: PARI/GP 2.15.4 (`gp -q`)
//!
//! GR(2^3, 2) = (Z/8)[x]/(x^2+x+1) and GR(3^3, 2) = (Z/27)[x]/(x^2+2x+2) (the Conway
//! polynomials of GF(4) and GF(9), which rustmath lifts), with w the class of x:
//!
//! ```text
//! D8 = Mod(1,8)*(x^2+x+1); w8 = Mod(Mod(1,8)*x, D8);
//! D27 = Mod(1,27)*(x^2+2*x+2); w27 = Mod(Mod(1,27)*x, D27);
//! lift(lift(w8+w8+1))                         \\ 2*x + 1
//! lift(lift(w8^3))                            \\ 1
//! lift(lift((1+2*w8)^2)), lift(lift((1+2*w8)^4))   \\ 5, 1
//! lift(lift(9*Mod(Mod(1,8),D8))), lift(lift(8*Mod(Mod(1,8),D8)))   \\ 1, 0
//! lift(lift(-Mod(Mod(1,8),D8)))               \\ 7
//! lift(lift(Mod(Mod(2,8),D8)^3))              \\ 0   (2 is nilpotent in GR(8,2))
//! znorder(Mod(2,27))                          \\ 18  (the order of 2 in GR(27,2))
//! lift(lift((y-w27)*(y+2+w27)))               \\ y^2 + 2*y + 2
//! f(t) = t^2+2*t+2; lift(lift(f(w27))), lift(lift(f(1+3*w27))), lift(lift(f(5*w27^0)))
//!                                             \\ 0, 21*x + 14, 10
//! lift(lift(w27 + 2*w27 + 5))                 \\ 3*x + 5
//! valuation(4, 2)                             \\ 2
//! Mod(-1,2) == Mod(1,2)                       \\ 1   (-1 = 1 in GR(2, d): order 1, not 2)
//! Vecrev((x+1)^8 - x^8)                       \\ [1, 8, 28, 56, 70, 56, 28, 8]
//! lift(lift(((x+1)^8 - x^8)*Mod(Mod(1,8),D8) + w8))   \\ 4*x^6 + 6*x^4 + 4*x^2 + (t + 1)
//! ```
//!
//! (In the last line the ring variable is `t`, i.e. `D8 = Mod(1,8)*(t^2+t+1)` and
//! `w8 = Mod(Mod(1,8)*t, D8)`, so that `x` can be the polynomial variable.)

use rustmath_core::divisibility::{FiniteChainRing, LocalRing};
use rustmath_core::valuation::DiscreteValuation;
use rustmath_core::{MathError, MultiplicativeOrder, Parent, Ring};
use rustmath_finitefields::{GaloisRing, GaloisRingElement};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;

type R = GaloisRingElement;

fn int(n: i64) -> Integer {
    Integer::from(n)
}

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&n| int(n)).collect()
}

/// GR(2^3, 2), H48E1 (#R = 64).
fn gr8() -> GaloisRing {
    GaloisRing::new(int(2), 3, 2).unwrap()
}

/// GR(3^3, 2).
fn gr27() -> GaloisRing {
    GaloisRing::new(int(3), 3, 2).unwrap()
}

/// Generic-code view: what `fn f<T: Ring>` sees.
fn hooks<T: Ring>(x: &T) -> (Option<bool>, Option<bool>, MultiplicativeOrder) {
    (
        x.unit_status(),
        x.nilpotent_status(),
        x.multiplicative_order_status(),
    )
}

/// A generic sum seeded with the static `T::zero()`.
fn generic_sum<T: Ring>(xs: &[T]) -> T {
    xs.iter().cloned().fold(T::zero(), |acc, x| acc + x)
}

/// A generic product seeded with the static `T::one()`.
fn generic_product<T: Ring>(xs: &[T]) -> T {
    xs.iter().cloned().fold(T::one(), |acc, x| acc * x)
}

/// The unbound constant `n` (a sum of `R::one()`s, as generic code builds it).
fn unbound(n: u32) -> R {
    (0..n).fold(R::zero(), |acc, _| acc + R::one())
}

#[test]
fn unbound_constants_bind_on_contact() {
    let r = gr8();
    let w = r.generator();
    assert!(!R::zero().is_bound() && !R::one().is_bound());
    assert_eq!(R::zero() + w.clone(), w);
    assert_eq!(w.clone() + R::zero(), w);
    assert_eq!(R::one() * w.clone(), w);
    assert_eq!(w.clone() - R::zero(), w);
    let z = R::zero() * w.clone();
    assert!(z.is_bound());
    assert_eq!(z.ring(), &r);
    assert_eq!(z, r.zero());
    assert_eq!(z.eltseq(), &ints(&[0, 0])[..]);

    // PARI: 9*1 = 1 and 8*1 = 0 in GR(8,2).
    let nine = unbound(9);
    assert!(!nine.is_bound());
    assert_eq!(nine.eltseq(), &[int(9)]);
    assert_eq!(nine.clone() * r.one(), r.one());
    assert_eq!(nine, r.one()); // coercing ==
    let eight = unbound(8);
    assert_eq!(eight.clone() * r.one(), r.zero());
    assert_eq!(eight, r.zero());
    // Both unbound: compared in Z (the documented caveat: 8 and 0 differ as integers).
    assert_ne!(eight, R::zero());
    assert!(!eight.is_zero());

    // Coercing == in both directions.
    assert_eq!(R::zero(), r.zero());
    assert_eq!(r.zero(), R::zero());
    assert_eq!(R::one(), r.one());
    assert_ne!(R::zero(), r.one());
    assert_ne!(r.one(), R::zero());

    // PARI: -1 = 7 in GR(8,2).
    assert_eq!((-R::one()) * r.one(), r.from_int(int(7)));
    assert_eq!(-R::one(), r.from_int(int(7)));

    // Bound elements of different rings are unequal, never a panic.
    assert_ne!(r.one(), gr27().one());
}

#[test]
fn generic_code_over_galois_rings() {
    let r = gr8();
    let w = r.generator();
    // PARI: w + w + 1 = 2x + 1, w^3 = 1.
    assert_eq!(
        generic_sum::<R>(&[w.clone(), w.clone(), r.one()]),
        r.element(ints(&[1, 2]))
    );
    assert_eq!(
        generic_product::<R>(&[w.clone(), w.clone(), w.clone()]),
        r.one()
    );
    // Empty sum and product: the unbound 0 and 1.
    assert!(generic_sum::<R>(&[]).is_zero());
    assert!(generic_product::<R>(&[]).is_one());
    // Ring::pow seeds with Self::one() for n = 0.
    assert_eq!(Ring::pow(&w, 0), r.one());
    // PARI: (1+2w)^2 = 5, (1+2w)^4 = 1.
    let u = r.element(ints(&[1, 2]));
    assert_eq!(Ring::pow(&u, 2), r.from_int(int(5)));
    assert_eq!(Ring::pow(&u, 4), r.one());
    // GR(27,2), PARI: w + 2w + 5 = 3x + 5.
    let r27 = gr27();
    let w27 = r27.generator();
    let s = generic_sum::<R>(&[w27.clone(), w27.clone() + w27.clone(), r27.from_int(int(5))]);
    assert_eq!(s, r27.element(ints(&[5, 3])));
    assert!(s.is_bound());
}

#[test]
fn polynomials_over_galois_ring() {
    let r = gr27();
    let w = r.generator();
    let c = |x: GaloisRingElement| UnivariatePolynomial::constant(x);
    // y from bound coefficients, so every coefficient of the product is bound.
    let y = UnivariatePolynomial::new(vec![r.zero(), r.one()]);
    // f = y^2 + 2y + 2.
    let f = UnivariatePolynomial::new(vec![r.from_int(int(2)), r.from_int(int(2)), r.one()]);
    // PARI: (y - w)(y + 2 + w) = y^2 + 2y + 2.
    let prod = (y.clone() - c(w.clone())) * (y + c(r.from_int(int(2)) + w.clone()));
    assert_eq!(prod, f);
    assert!(prod.coefficients().iter().all(|a| a.is_bound()));
    // The same product with the static `var()` (whose coefficients are unbound 0 and 1): the
    // leading coefficient stays unbound(1), which equals r.one() under the coercing ==.
    let x = UnivariatePolynomial::<R>::var();
    let prod2 = (x.clone() - c(w.clone())) * (x + c(r.from_int(int(2)) + w.clone()));
    assert_eq!(prod2, f);
    // PARI: f(w) = 0, f(1 + 3w) = 14 + 21w, f(5) = 10.
    assert!(f.evaluate(&w).is_zero());
    let one_3w = r.element(ints(&[1, 3]));
    assert_eq!(f.evaluate(&one_3w), r.element(ints(&[14, 21])));
    assert_eq!(f.evaluate(&r.from_int(int(5))), r.from_int(int(10)));
    // Coefficient past the degree (MAGMA `Coefficient(p, i)` is 0 there).
    assert_eq!(f.coeff(5), &r.zero());
    assert!(f.coeff(5).is_zero());
}

/// The containers of rustmath-polynomials bind unbound constants (`Ring::bind_to_parent_of`):
/// a polynomial built from the static `var()` and `R::one()` is bound as soon as one
/// coefficient is, so degrees, zero tests and `==` follow GR(8, 2). (This replaces the test
/// that pinned the old caveat, degree 7: gp `lift(Mod(1,8)*((x+1)^8-x^8))` =
/// `4*x^6 + 6*x^4 + 4*x^2 + 1` shows the degree is 6 in GR(8, 2)[x].) A polynomial with only
/// unbound coefficients stays the image of an element of Z[x].
#[test]
fn polynomials_from_static_constructors_bind_on_construction() {
    type P = UnivariatePolynomial<R>;
    let r = gr8();
    let w = r.generator();
    let x8 = |x: &P| {
        let mut p = P::constant(R::one());
        for _ in 0..8 {
            p = p * x.clone();
        }
        p
    };

    // q = (x + 1)^8 - x^8 + w from the static constructors.
    let x = P::var();
    let one = P::constant(R::one());
    let unbound = x8(&(x.clone() + one)) - x8(&x);
    // PARI: Vecrev((x+1)^8 - x^8) = [1, 8, 28, 56, 70, 56, 28, 8], all unbound (in Z[x]).
    assert_eq!(unbound.degree(), Some(7));
    for (i, c) in [1, 8, 28, 56, 70, 56, 28, 8].into_iter().enumerate() {
        let coeff = unbound.coeff(i);
        assert!(!coeff.is_bound(), "coefficient {i} should be unbound");
        assert_eq!(coeff.eltseq(), &[int(c)][..]);
    }
    let q = unbound + P::constant(w.clone());

    // The same polynomial from bound elements. PARI: (1 + w) + 4x^2 + 6x^4 + 4x^6.
    let y = P::new(vec![r.zero(), r.one()]);
    let one_b = P::constant(r.one());
    let qb = x8(&(y.clone() + one_b)) - x8(&y) + P::constant(w.clone());
    let want = P::new(vec![
        r.one() + w.clone(),
        r.zero(),
        r.from_int(int(4)),
        r.zero(),
        r.from_int(int(6)),
        r.zero(),
        r.from_int(int(4)),
    ]);
    assert_eq!(qb.degree(), Some(6));
    assert_eq!(qb, want);
    assert!(qb.coefficients().iter().all(|c| c.is_bound()));

    // q is bound on construction: degree 6, every coefficient bound, equal to qb.
    assert_eq!(q.degree(), Some(6));
    assert!(q.coefficients().iter().all(|c| c.is_bound()));
    assert_eq!(q, qb);
    assert_eq!(q.coeff(0), &(r.one() + w.clone()));
    // A coefficient past the degree is the bound zero.
    assert!(q.coeff(9).is_bound());
    assert!(q.coeff(9).is_zero());
    // Binding with scalar_mul still agrees.
    assert_eq!(q.scalar_mul(&r.one()), qb);

    // Equality coerces across lengths: 1 + 8x is 1 in GR(8, 2)[x] (but not in Z[x]).
    let eight = (0..8).fold(R::zero(), |acc, _| acc + R::one());
    let u = P::new(vec![R::one(), eight]);
    assert_eq!(u.degree(), Some(1));
    assert_eq!(u, P::constant(r.one()));
    assert_eq!(P::constant(r.one()), u);
    assert_ne!(u, P::constant(R::one()));
}

/// The sparse containers bind in `add_term`, and drop the terms that become zero.
/// gp: lift(Mod(1,8)*((x+1)^8-x^8)) = 4*x^6 + 6*x^4 + 4*x^2 + 1, so over GR(8, 2)
/// (x_0 + 1)^8 - x_0^8 + w = (1 + w) + 4x_0^2 + 6x_0^4 + 4x_0^6 (4 terms; the old code kept 8,
/// degree 7), and with q^-1 for x the Laurent value is (1 + w) + 4q^-2 + 6q^-4 + 4q^-6.
#[test]
fn sparse_polynomials_from_static_constructors_bind() {
    use rustmath_polynomials::{LaurentPolynomial, Monomial, MultivariatePolynomial};
    type M = MultivariatePolynomial<R>;
    type L = LaurentPolynomial<R>;
    let r = gr8();
    let w = r.generator();
    let pow_m = |x: &M, k: u32| {
        let mut p = M::constant(R::one());
        for _ in 0..k {
            p = p * x.clone();
        }
        p
    };
    let x = M::variable(0);
    let unbound = pow_m(&(x.clone() + M::constant(R::one())), 8) - pow_m(&x, 8);
    assert_eq!(unbound.num_terms(), 8);
    for q in [
        unbound.clone() + M::constant(w.clone()),
        M::constant(w.clone()) + unbound.clone(),
    ] {
        assert_eq!(q.num_terms(), 4);
        assert_eq!(q.degree(), Some(6));
        assert!(q.terms().all(|(_, c)| c.is_bound()));
        let coeff = |k: u32| q.coefficient(&Monomial::variable(0, k));
        assert_eq!(coeff(0).eltseq(), &ints(&[1, 1])[..]);
        assert_eq!(coeff(2), r.from_int(int(4)));
        assert_eq!(coeff(4), r.from_int(int(6)));
        assert_eq!(coeff(6), r.from_int(int(4)));
        for k in [1, 3, 5, 7] {
            assert!(coeff(k).is_zero() && coeff(k).is_bound(), "x_0^{k} is absent");
        }
        // Equal to the same value built from r.one().
        let xb = M::variable(0) * M::constant(r.one());
        let qb = pow_m(&(xb.clone() + M::constant(r.one())), 8) - pow_m(&xb, 8)
            + M::constant(w.clone());
        assert_eq!(qb.num_terms(), 4);
        assert_eq!(q, qb);
    }

    let qi = L::monomial(R::one(), -1);
    let mut lp = L::constant(R::one());
    let mut li = L::constant(R::one());
    for _ in 0..8 {
        lp = lp * (qi.clone() + L::constant(R::one()));
        li = li * qi.clone();
    }
    let l = lp - li + L::constant(w.clone());
    let terms = l.terms();
    assert_eq!(terms.len(), 4);
    assert_eq!(terms[0].0, -6);
    assert_eq!(terms[0].1, r.from_int(int(4)));
    assert_eq!(terms[1].0, -4);
    assert_eq!(terms[1].1, r.from_int(int(6)));
    assert_eq!(terms[2].0, -2);
    assert_eq!(terms[2].1, r.from_int(int(4)));
    assert_eq!(terms[3].0, 0);
    assert_eq!(terms[3].1, r.one() + w);
    assert!(terms.iter().all(|(_, c)| c.is_bound()));
}

#[test]
fn unbound_ring_hooks() {
    use MultiplicativeOrder::{Finite, NotAUnit, Undecided};
    assert_eq!(hooks(&R::one()), (Some(true), Some(false), Finite(1)));
    // -1 is a unit; its order is 1 in GR(2, d) (PARI Mod(-1,2) == Mod(1,2)) and 2 elsewhere.
    assert_eq!(hooks(&-R::one()), (Some(true), Some(false), Undecided));
    assert_eq!(hooks(&R::zero()), (Some(false), Some(true), NotAUnit));
    // 2 is a unit in GR(27, 2) and nilpotent in GR(8, 2): undecided.
    let two = R::one() + R::one();
    assert_eq!(hooks(&two), (None, None, Undecided));
    assert!(matches!(
        Ring::unit_inverse(&two),
        Err(MathError::InvalidArgument(_))
    ));
    assert_eq!(Ring::unit_inverse(&-R::one()), Ok(-R::one()));
    assert_eq!(
        Ring::unit_inverse(&R::zero()),
        Err(MathError::NotInvertible)
    );
    assert_eq!(GaloisRingElement::unit_inverse(&R::one()), Ok(R::one()));

    // Bound into a ring, 2 is decided. PARI: Mod(2,8)^3 = 0; znorder(Mod(2,27)) = 18.
    let r8 = gr8();
    let r27 = gr27();
    assert!((two.clone() * r8.one()).is_nilpotent());
    assert_eq!(
        hooks(&(two.clone() * r8.one())),
        (Some(false), Some(true), NotAUnit)
    );
    assert_eq!(
        (two.clone() * r27.one()).multiplicative_order(),
        Some(int(18))
    );
    assert_eq!(
        hooks(&(two * r27.one())),
        (Some(true), Some(false), Finite(18))
    );

    // Inherent predicates on the decided constants.
    assert!(R::one().is_unit());
    assert!((-R::one()).is_unit());
    assert!(!R::zero().is_unit());
    assert!(R::zero().is_nilpotent());
    assert!(!R::one().is_nilpotent());
    assert!(!R::zero().is_zero_divisor());
    assert!(!R::one().is_zero_divisor());
    assert!(R::zero().is_idempotent() && R::one().is_idempotent());
    assert_eq!(R::one().multiplicative_order(), Some(int(1)));
    assert_eq!(R::zero().multiplicative_order(), None);
    assert!(LocalRing::is_unit(&R::one()));
    assert!(!LocalRing::is_unit(&R::zero()));
}

#[test]
fn parent_and_valuation_with_unbound() {
    let r = gr8();
    assert!(!r.contains(&R::zero()));
    assert!(r.contains(&r.zero()));
    assert!(r.contains(&(R::zero() + r.one())));
    // The parent is known: the unbound element is bound first. PARI valuation(4,2) = 2;
    // 8 = 0 in Z/8, valuation infinity.
    assert_eq!(DiscreteValuation::valuation(&r, &unbound(4)), 2);
    assert_eq!(DiscreteValuation::valuation(&r, &unbound(8)), i64::MAX);
    assert_eq!(DiscreteValuation::valuation(&r, &R::one()), 0);
    // GCD = p^min(v(0) = a, v(2) = 1) = 2; LCM with 0 is 0.
    let two = r.from_int(int(2));
    assert_eq!(R::zero().gcd(&two), two);
    assert_eq!(two.gcd(&R::zero()), two);
    assert_eq!(R::zero().lcm(&two), r.zero());
    // v(1) = 0 < v(2) = 1: quotient 0, remainder 1.
    assert_eq!(R::one().quotrem(&two), Ok((r.zero(), r.one())));
    assert_eq!(R::one().rem_euclid(&two), Ok(r.one()));
    // xgcd(1, 2): g = 1 = 1*1 + 0*2.
    let (g, s, t) = R::one().xgcd(&two).unwrap();
    assert_eq!(g, r.one());
    assert_eq!(s * r.one() + t * two.clone(), g);
    // Both unbound: the answer depends on the ring.
    assert!(matches!(
        R::one().quotrem(&R::one()),
        Err(MathError::InvalidArgument(_))
    ));
    assert!(matches!(
        R::one().xgcd(&R::zero()),
        Err(MathError::InvalidArgument(_))
    ));
}

// The static constructors give different values, as they must: no Galois ring is the zero
// ring (#GR(p^a, d) = p^(ad) >= 2; MAGMA H48E1: #GR(8, 2) = 64), so 0 != 1 in each of them.
#[test]
fn static_zero_and_one_differ() {
    assert_ne!(<R as Ring>::zero(), <R as Ring>::one());
    assert!(<R as Ring>::zero().is_zero() && !<R as Ring>::zero().is_one());
    assert!(<R as Ring>::one().is_one() && !<R as Ring>::one().is_zero());
}

#[test]
#[should_panic(expected = "unbound")]
fn unbound_has_no_ring() {
    let _ = R::zero().ring();
}

#[test]
#[should_panic(expected = "unbound")]
fn unbound_has_no_valuation() {
    let _ = R::one().valuation();
}

// 2 is a unit in GR(27, 2) and nilpotent in GR(8, 2).
#[test]
#[should_panic(expected = "unbound")]
fn unbound_two_is_unit_is_undecided() {
    let _ = (R::one() + R::one()).is_unit();
}

// The `LocalRing` trait method delegates to the inherent `is_unit`, so it panics too (the
// `Ring::unit_status` hook answers `None` instead; see `unbound_ring_hooks`).
#[test]
#[should_panic(expected = "unbound")]
fn unbound_two_local_ring_is_unit_is_undecided() {
    let _ = LocalRing::is_unit(&(R::one() + R::one()));
}

// -1 has order 1 in GR(2, d) and 2 elsewhere.
#[test]
#[should_panic(expected = "unbound")]
fn unbound_minus_one_order_is_undecided() {
    let _ = (-R::one()).multiplicative_order();
}

#[test]
#[should_panic(expected = "unbound")]
fn unbound_gcd_of_two_constants_is_refused() {
    let _ = R::one().gcd(&R::zero());
}

#[test]
#[should_panic(expected = "unbound")]
fn unbound_has_no_nilpotency_index() {
    let _ = FiniteChainRing::nilpotency_index(&R::one());
}

#[test]
#[should_panic(expected = "unbound")]
fn unbound_residue_needs_the_ring() {
    let _ = R::one().residue();
}

// The cross-ring panic is unchanged.
#[test]
#[should_panic(expected = "different Galois rings")]
fn bound_elements_of_different_rings_do_not_mix() {
    let _ = gr8().one() + gr27().one();
}
