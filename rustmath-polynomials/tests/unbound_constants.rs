//! Containers bind unbound integer constants (RustMath; MAGMA has none). The parent-carrying
//! element types of rustmath-finitefields return from the static `zero()`/`one()` an integer
//! constant c that awaits the canonical map Z -> R (MAGMA Handbook ch. 48 §48.2.3: `R ! a` is
//! "a mod p^a"; digest `rustmath-lava/docs/Chapter_48_Galois_Rings.md`). The polynomial
//! containers of ch. 23 (digest `Chapter_23_Univariate_Polynomial_Rings.md`) bind such
//! constants to the parent of a bound coefficient with the `Ring` hooks
//! `HAS_UNBOUND_CONSTANTS`, `is_unbound_constant` and `bind_to_parent_of`. Binding is the
//! unique ring map Z -> R, so it does not change the element of R[x] a polynomial stands for.
//!
//! The coefficient type here is a test sentinel `Zs` with the same rules as the modulus-0
//! `IntegerMod` sentinel. Reference values, PARI/GP 2.15.4 (`gp -q`):
//! ```text
//! lift(Mod(1,8)*((x+1)^8 - x^8 + 1))      \\ 4*x^6 + 6*x^4 + 4*x^2 + 2
//! Vecrev((x+1)^8 - x^8)                   \\ [1, 8, 28, 56, 70, 56, 28, 8]
//! lift(Mod(1,8)*((x+1)^4*y + 3*y^2))      \\ y*x^4 + 4*y*x^3 + 6*y*x^2 + 4*y*x + (3*y^2 + y)
//! subst(lift(Mod(1,8)*((y+1)^8 - y^8 + 1)), y, 1/q)   \\ (2*q^6 + 4*q^4 + 6*q^2 + 4)/q^6
//! ```

use rustmath_core::Ring;
use rustmath_integers::Integer;
use rustmath_polynomials::{
    LaurentPolynomial, Monomial, MultivariatePolynomial, UnivariatePolynomial,
};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Z/m with an unbound sentinel: `m = 0` is an integer constant in Z awaiting the canonical
/// map. Arithmetic binds an unbound operand into the other operand's modulus and reduces;
/// two bound operands with different moduli panic; `==` coerces the same way. `zero()` and
/// `one()` are unbound.
#[derive(Clone, Copy, Debug)]
struct Zs {
    v: i64,
    m: i64,
}

impl Zs {
    fn bound(v: i64, m: i64) -> Self {
        assert!(m >= 2);
        Zs {
            v: v.rem_euclid(m),
            m,
        }
    }
    fn unbound(v: i64) -> Self {
        Zs { v, m: 0 }
    }
    fn make(v: i64, m: i64) -> Self {
        if m == 0 {
            Zs::unbound(v)
        } else {
            Zs::bound(v, m)
        }
    }
    fn common(&self, other: &Self) -> i64 {
        match (self.m, other.m) {
            (0, m) | (m, 0) => m,
            (a, b) => {
                assert_eq!(a, b, "Zs: different moduli");
                a
            }
        }
    }
}

impl PartialEq for Zs {
    fn eq(&self, other: &Self) -> bool {
        match (self.m, other.m) {
            (0, 0) => self.v == other.v,
            (0, m) => self.v.rem_euclid(m) == other.v,
            (m, 0) => other.v.rem_euclid(m) == self.v,
            (a, b) => a == b && self.v == other.v,
        }
    }
}

impl fmt::Display for Zs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.m == 0 {
            write!(f, "{}", self.v)
        } else {
            write!(f, "{} mod {}", self.v, self.m)
        }
    }
}

impl Add for Zs {
    type Output = Zs;
    fn add(self, o: Zs) -> Zs {
        Zs::make(self.v + o.v, self.common(&o))
    }
}
impl Sub for Zs {
    type Output = Zs;
    fn sub(self, o: Zs) -> Zs {
        Zs::make(self.v - o.v, self.common(&o))
    }
}
impl Mul for Zs {
    type Output = Zs;
    fn mul(self, o: Zs) -> Zs {
        Zs::make(self.v * o.v, self.common(&o))
    }
}
impl Neg for Zs {
    type Output = Zs;
    fn neg(self) -> Zs {
        Zs::make(-self.v, self.m)
    }
}

impl Ring for Zs {
    fn zero() -> Self {
        Zs::unbound(0)
    }
    fn one() -> Self {
        Zs::unbound(1)
    }
    fn is_zero(&self) -> bool {
        self.v == 0
    }
    fn is_one(&self) -> bool {
        self.v == 1
    }
    const HAS_UNBOUND_CONSTANTS: bool = true;
    fn is_unbound_constant(&self) -> bool {
        self.m == 0
    }
    fn bind_to_parent_of(&self, model: &Self) -> Option<Self> {
        (self.m == 0 && model.m != 0).then(|| Zs::bound(self.v, model.m))
    }
}

type P = UnivariatePolynomial<Zs>;
type MP = MultivariatePolynomial<Zs>;
type LP = LaurentPolynomial<Zs>;

fn values(p: &P) -> Vec<(i64, i64)> {
    p.coefficients().iter().map(|c| (c.v, c.m)).collect()
}

fn pow_mp(base: &MP, k: u32) -> MP {
    let mut acc = MP::constant(Zs::one());
    for _ in 0..k {
        acc = acc * base.clone();
    }
    acc
}

fn pow_lp(base: &LP, k: u32) -> LP {
    let mut acc = LP::constant(Zs::one());
    for _ in 0..k {
        acc = acc * base.clone();
    }
    acc
}

/// (x + 1)^8 - x^8 from `var()` and `R::one()` only: every coefficient is unbound.
fn unbound_binomial() -> P {
    let x = P::var();
    let one = P::constant(Zs::one());
    Ring::pow(&(x.clone() + one), 8) - Ring::pow(&x, 8)
}

#[test]
fn univariate_binds_mixed_coefficients() {
    // gp: lift(Mod(1,8)*((x+1)^8 - x^8 + 1)) = 4*x^6 + 6*x^4 + 4*x^2 + 2.
    let q = unbound_binomial() + P::constant(Zs::bound(1, 8));
    assert_eq!(q.degree(), Some(6));
    assert_eq!(
        values(&q),
        vec![(2, 8), (0, 8), (4, 8), (0, 8), (6, 8), (0, 8), (4, 8)]
    );
    assert_eq!(q.coeff(9).m, 8);
    assert!(q.coeff(9).is_zero());
    assert!(!q.is_unbound_constant());
    // The same with the bound constant added first.
    let q2 = P::constant(Zs::bound(1, 8)) + unbound_binomial();
    assert_eq!(values(&q2), values(&q));
}

#[test]
fn univariate_all_unbound_stays_in_z() {
    // gp: Vecrev((x+1)^8 - x^8) = [1, 8, 28, 56, 70, 56, 28, 8].
    let q = unbound_binomial();
    assert_eq!(q.degree(), Some(7));
    assert_eq!(
        values(&q),
        vec![
            (1, 0),
            (8, 0),
            (28, 0),
            (56, 0),
            (70, 0),
            (56, 0),
            (28, 0),
            (8, 0)
        ]
    );
    assert!(q.is_unbound_constant());
    assert_eq!(q.coeff(9).m, 0);
}

#[test]
fn univariate_eq_coerces_across_lengths() {
    // 1 + 8x is 1 in (Z/8)[x], but not in Z[x].
    let u = P::new(vec![Zs::unbound(1), Zs::unbound(8)]);
    let b = P::constant(Zs::bound(1, 8));
    assert_eq!(u.degree(), Some(1));
    assert!(u == b);
    assert!(b == u);
    assert!(u != P::constant(Zs::unbound(1)));
    assert!(P::constant(Zs::unbound(1)) != u);
    // Two bound polynomials of different degrees stay unequal.
    assert!(b != P::new(vec![Zs::bound(1, 8), Zs::bound(1, 8)]));
    // Binding can shorten the unbound side to a different degree: 1 + 8x + x^2 is not 1.
    let u2 = P::new(vec![Zs::unbound(1), Zs::unbound(8), Zs::unbound(1)]);
    assert!(u2 != b && b != u2);
}

#[test]
fn nested_univariate_binds() {
    // gp: lift(Mod(1,8)*((x+1)^4*y + 3*y^2)) = y*x^4 + 4*y*x^3 + 6*y*x^2 + 4*y*x + (3*y^2 + y).
    type Q = UnivariatePolynomial<P>;
    let x = P::var();
    let a = Ring::pow(&(x + P::constant(Zs::one())), 4);
    assert!(a.is_unbound_constant());
    let b = P::constant(Zs::bound(3, 8));
    let y = Q::var();
    let f = Q::constant(a) * y.clone() + Q::constant(b) * y.clone() * y;
    assert_eq!(f.degree(), Some(2));
    assert!(!f.is_unbound_constant());
    let c = f.coefficients();
    assert_eq!(values(&c[0]), vec![(0, 8)]);
    assert_eq!(values(&c[1]), vec![(1, 8), (4, 8), (6, 8), (4, 8), (1, 8)]);
    assert_eq!(values(&c[2]), vec![(3, 8)]);
    // A missing coefficient past the degree is bound too.
    assert_eq!(values(f.coeff(5)), vec![(0, 8)]);
}

#[test]
fn univariate_over_laurent_binds() {
    type Q = UnivariatePolynomial<LP>;
    let f = Q::new(vec![LP::new(), LP::var(), LP::constant(Zs::bound(3, 8))]);
    assert!(!f.is_unbound_constant());
    let c = f.coefficients();
    assert!(c[0].is_zero());
    assert!(c[0].terms().is_empty());
    assert_eq!(
        c[1].terms()
            .iter()
            .map(|(e, c)| (*e, c.v, c.m))
            .collect::<Vec<_>>(),
        vec![(1, 1, 8)]
    );
    let g = Q::new(vec![
        LP::new(),
        LP::monomial(Zs::bound(1, 8), 1),
        LP::constant(Zs::bound(3, 8)),
    ]);
    assert!(f == g);
    assert!(g == f);
}

/// The terms of a multivariate polynomial in x_0 as (exponent, value, modulus).
fn mp_terms(p: &MP) -> Vec<(u32, i64, i64)> {
    p.terms().map(|(m, c)| (m.exponent(0), c.v, c.m)).collect()
}

#[test]
fn multivariate_binds_on_add_term() {
    // gp: lift(Mod(1,8)*((x+1)^8 - x^8 + 1)) = 4*x^6 + 6*x^4 + 4*x^2 + 2.
    let x = MP::variable(0);
    let one = MP::constant(Zs::one());
    let unbound = pow_mp(&(x.clone() + one), 8) - pow_mp(&x, 8);
    assert_eq!(unbound.num_terms(), 8);
    assert_eq!(unbound.degree(), Some(7));
    let q = unbound.clone() + MP::constant(Zs::bound(1, 8));
    let q2 = MP::constant(Zs::bound(1, 8)) + unbound;
    let expected = vec![(0, 2, 8), (2, 4, 8), (4, 6, 8), (6, 4, 8)];
    let mut got = mp_terms(&q);
    got.sort();
    assert_eq!(got, expected);
    let mut got2 = mp_terms(&q2);
    got2.sort();
    assert_eq!(got2, expected);
    assert_eq!(q.num_terms(), 4);
    assert_eq!(q.degree(), Some(6));
    let c3 = q.coefficient(&Monomial::variable(0, 3));
    assert!(c3.is_zero());
    assert_eq!(c3.m, 8);
    assert!(q == q2);
}

#[test]
fn multivariate_eq_coerces() {
    let mut u = MP::zero();
    u.add_term(Monomial::new(), Zs::one());
    u.add_term(Monomial::variable(0, 1), Zs::unbound(8));
    assert_eq!(u.num_terms(), 2);
    let b = MP::constant(Zs::bound(1, 8));
    assert!(u == b);
    assert!(b == u);
    assert!(u != MP::constant(Zs::unbound(1)));
    assert!(MP::constant(Zs::unbound(1)) != u);
    // A sparse zero carries no parent: the unbound 8 is a nonzero element of Z[x].
    assert!(MP::constant(Zs::unbound(8)) != MP::zero());
}

#[test]
fn laurent_binds_on_add_term() {
    // gp: subst(lift(Mod(1,8)*((y+1)^8 - y^8 + 1)), y, 1/q) = (2*q^6 + 4*q^4 + 6*q^2 + 4)/q^6.
    let qi = LP::monomial(Zs::one(), -1);
    let one = LP::constant(Zs::one());
    let unbound = pow_lp(&(qi.clone() + one), 8) - pow_lp(&qi, 8);
    assert_eq!(unbound.terms().len(), 8);
    assert_eq!(unbound.min_exponent(), Some(-7));
    let f = unbound + LP::constant(Zs::bound(1, 8));
    let got: Vec<(i32, i64, i64)> = f.terms().iter().map(|(e, c)| (*e, c.v, c.m)).collect();
    assert_eq!(got, vec![(-6, 4, 8), (-4, 6, 8), (-2, 4, 8), (0, 2, 8)]);
    let c = f.coeff(-3);
    assert!(c.is_zero());
    assert_eq!(c.m, 8);
}

#[test]
fn laurent_ring_hooks() {
    const { assert!(<LP as Ring>::HAS_UNBOUND_CONSTANTS) };
    const { assert!(!<LaurentPolynomial<Integer> as Ring>::HAS_UNBOUND_CONSTANTS) };
    assert!(<LP as Ring>::one().is_unbound_constant());
    assert!(<LP as Ring>::zero().is_unbound_constant());
    assert!(LP::var().is_unbound_constant());
    let bound = LP::constant(Zs::bound(3, 8));
    assert!(!bound.is_unbound_constant());
    let v = LP::var()
        .bind_to_parent_of(&bound)
        .expect("unbound into bound");
    assert_eq!(
        v.terms()
            .iter()
            .map(|(e, c)| (*e, c.v, c.m))
            .collect::<Vec<_>>(),
        vec![(1, 1, 8)]
    );
    assert!(bound.bind_to_parent_of(&LP::var()).is_none());
    assert!(LP::var().bind_to_parent_of(&LP::var()).is_none());
    // An unbound term that is 0 in the bound ring is dropped.
    let eight = LP::monomial(Zs::unbound(8), 2);
    assert!(eight.bind_to_parent_of(&bound).expect("binds").is_zero());
}

#[test]
fn default_hooks_are_inert_for_containers() {
    const { assert!(!<UnivariatePolynomial<Integer> as Ring>::HAS_UNBOUND_CONSTANTS) };
    const { assert!(!<LaurentPolynomial<Integer> as Ring>::HAS_UNBOUND_CONSTANTS) };
    let x = UnivariatePolynomial::<Integer>::var();
    let one = UnivariatePolynomial::<Integer>::constant(Integer::from(1));
    assert!(!x.is_unbound_constant());
    assert!(x.bind_to_parent_of(&one).is_none());
    assert!(x.clone() + one.clone() != x);
    assert!(x.clone() + one.clone() == one + x);

    let l = LaurentPolynomial::<Integer>::var();
    assert!(!l.is_unbound_constant());
    assert!(l
        .bind_to_parent_of(&LaurentPolynomial::constant(Integer::from(2)))
        .is_none());
    assert!(l.clone() + LaurentPolynomial::constant(Integer::from(1)) != l);

    let m = MultivariatePolynomial::<Integer>::variable(0);
    let mone = MultivariatePolynomial::constant(Integer::from(1));
    assert!(m.clone() + mone.clone() != m);
    assert!(m.clone() + mone.clone() == mone + m);
}
