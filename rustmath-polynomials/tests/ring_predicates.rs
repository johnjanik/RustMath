//! The `Ring` hooks of `UnivariatePolynomial` and `LaurentPolynomial` (MAGMA Handbook ch. 23
//! §23.4.4 `IsUnit`, `IsNilpotent`; ch. 17 §17.4.2 `Characteristic`, §17.4.3
//! `IsCommutative`; digest `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`).
//! Rules: every ring (`f ↦ f(0)`, unit leading coefficients), and over a commutative
//! coefficient ring Atiyah-Macdonald, *Introduction to Commutative Algebra*, Ch. 1, Ex. 2.
//! The inverse is the power-series recurrence of `1/f` under a work bound, checked on both
//! sides.
//!
//! Reference values, GAP 4 (`gap -q`):
//! ```text
//! R := PolynomialRing(Integers, ["x"]);; x := IndeterminatesOfPolynomialRing(R)[1];;
//! IsUnit(R, 2*x+1); IsUnit(R, 2*x^2-1); IsUnit(R, x); IsUnit(R, -One(R));   # false false false true
//! Q := PolynomialRing(Rationals, ["y"]);; y := IndeterminatesOfPolynomialRing(Q)[1];;
//! IsUnit(Q, y+1); Inverse(2*One(Q));                                       # false, 1/2
//! S := PolynomialRing(ZmodnZ(8), ["t"]);; t := IndeterminatesOfPolynomialRing(S)[1];;
//! List([1..4], k -> (One(S)+2*t)^k);       # 1+2t, 1+4t+4t^2, 1+6t+4t^2, 1: order 4
//! (One(S)+2*t)*(One(S)+6*t+4*t^2);         # 1 (both orders)
//! (One(S)+4*t+2*t^2)*(One(S)+4*t+6*t^2+4*t^4);  # 1
//! (2*One(S)+4*t)^3;                        # 0
//! T := PolynomialRing(ZmodnZ(12), ["u"]);; u := IndeterminatesOfPolynomialRing(T)[1];;
//! List([1..2], k -> (5*One(T)+6*u)^k);     # 5+6u, 1: order 2
//! (6*One(T)+6*u^2)^2;                      # 0
//! List([1..4], k -> 4^k mod 12);           # [4,4,4,4]: 4 is not nilpotent mod 12
//! P := PolynomialRing(ZmodnZ(2^62), ["s"]);; s := IndeterminatesOfPolynomialRing(P)[1];;
//! (One(P)+2*s) * Sum([0..61], k -> ((-2)^k mod 2^62) * s^k);   # 1
//! A := [[0,1],[1,0]];; F := [[1,0],[0,1]] + A*x + [[1,0],[0,0]]*x^2;;
//! H := [[1,0],[0,1]] - A*x + [[0,0],[0,1]]*x^2;; F*H; H*F;      # both the identity
//! DeterminantMat([[1+x, 0*x], [0*x, One(R)]]);                  # x+1: I + E11 x is no unit
//! Y := PolynomialRing(ZmodnZ(6), ["y"]);; y := IndeterminatesOfPolynomialRing(Y)[1];;
//! (3*One(Y)+4*y)*(4*One(Y)+3*y);           # y, so (3+4x)(3+4x^-1) = 1 over Z/6
//! ```
//! Sage docstrings (`polynomial_element.pyx`, `is_nilpotent`): over `Integers(12)`, `5 + 6x`
//! is not nilpotent and `6 + 6x^2` is.

use rustmath_core::{MathError, MultiplicativeOrder, Ring};
use rustmath_integers::Integer;
use rustmath_polynomials::{LaurentPolynomial, UnivariatePolynomial};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------------------
// Test coefficient rings.

/// Z/M with exact hooks. `COMM` sets `static_is_commutative` (`Some(true)` or the default
/// `None`), to test the rules that need commutativity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Zm<const M: u64, const COMM: bool>(u64);

type Z12 = Zm<12, true>;
type Z2e62 = Zm<{ 1 << 62 }, true>;
type Z8NoFlag = Zm<8, false>;

impl<const M: u64, const C: bool> Zm<M, C> {
    fn new(a: i64) -> Self {
        Zm(a.rem_euclid(M as i64) as u64)
    }
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
}

impl<const M: u64, const C: bool> fmt::Display for Zm<M, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<const M: u64, const C: bool> Add for Zm<M, C> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Zm(((self.0 as u128 + o.0 as u128) % M as u128) as u64)
    }
}
impl<const M: u64, const C: bool> Sub for Zm<M, C> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Zm(((self.0 as u128 + M as u128 - o.0 as u128) % M as u128) as u64)
    }
}
impl<const M: u64, const C: bool> Mul for Zm<M, C> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Zm(((self.0 as u128 * o.0 as u128) % M as u128) as u64)
    }
}
impl<const M: u64, const C: bool> Neg for Zm<M, C> {
    type Output = Self;
    fn neg(self) -> Self {
        Zm((M - self.0) % M)
    }
}
impl<const M: u64, const C: bool> Ring for Zm<M, C> {
    fn zero() -> Self {
        Zm(0)
    }
    fn one() -> Self {
        Zm(1 % M)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn is_one(&self) -> bool {
        self.0 == 1 % M
    }
    fn unit_status(&self) -> Option<bool> {
        Some(Self::gcd(self.0, M) == 1)
    }
    /// a is nilpotent mod M iff a^k = 0 for some k <= 64 (M < 2^64).
    fn nilpotent_status(&self) -> Option<bool> {
        let mut p = *self;
        for _ in 0..64 {
            if p.is_zero() {
                return Some(true);
            }
            p = p * *self;
        }
        Some(false)
    }
    fn unit_inverse(&self) -> rustmath_core::Result<Self> {
        let (mut r0, mut r1) = (M as i128, self.0 as i128);
        let (mut s0, mut s1) = (0i128, 1i128);
        while r1 != 0 {
            let q = r0 / r1;
            (r0, r1) = (r1, r0 - q * r1);
            (s0, s1) = (s1, s0 - q * s1);
        }
        if r0 != 1 {
            return Err(MathError::NotInvertible);
        }
        Ok(Zm(s0.rem_euclid(M as i128) as u64))
    }
    fn static_characteristic() -> Option<u64> {
        Some(M)
    }
    fn static_is_commutative() -> Option<bool> {
        if C {
            Some(true)
        } else {
            None
        }
    }
}

/// 2×2 integer matrices, row major. `unit_status` by `det = ±1`, `nilpotent_status` by
/// `A^2 = 0` (a 2×2 matrix is nilpotent iff its square is 0), default commutativity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct M2([i64; 4]);

const I2: M2 = M2([1, 0, 0, 1]);
const E11: M2 = M2([1, 0, 0, 0]);
const E12: M2 = M2([0, 1, 0, 0]);
const E22: M2 = M2([0, 0, 0, 1]);
const SWAP: M2 = M2([0, 1, 1, 0]);

impl M2 {
    fn det(&self) -> i64 {
        self.0[0] * self.0[3] - self.0[1] * self.0[2]
    }
}
impl fmt::Display for M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}
impl Add for M2 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        M2(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }
}
impl Sub for M2 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        M2(std::array::from_fn(|i| self.0[i] - o.0[i]))
    }
}
impl Mul for M2 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let [a, b, c, d] = self.0;
        let [e, f, g, h] = o.0;
        M2([a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h])
    }
}
impl Neg for M2 {
    type Output = Self;
    fn neg(self) -> Self {
        M2(self.0.map(|x| -x))
    }
}
impl Ring for M2 {
    fn zero() -> Self {
        M2([0; 4])
    }
    fn one() -> Self {
        I2
    }
    fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }
    fn is_one(&self) -> bool {
        *self == I2
    }
    fn unit_status(&self) -> Option<bool> {
        Some(self.det().abs() == 1)
    }
    fn nilpotent_status(&self) -> Option<bool> {
        Some((*self * *self).is_zero())
    }
    fn unit_inverse(&self) -> rustmath_core::Result<Self> {
        let d = self.det();
        if d.abs() != 1 {
            return Err(MathError::NotInvertible);
        }
        let [a, b, c, e] = self.0;
        Ok(M2([e * d, -b * d, -c * d, a * d]))
    }
}

// ---------------------------------------------------------------------------------------
// Helpers.

fn zx(c: &[i64]) -> UnivariatePolynomial<Integer> {
    UnivariatePolynomial::new(c.iter().map(|&a| Integer::from(a)).collect())
}

fn zm<const M: u64, const C: bool>(c: &[i64]) -> UnivariatePolynomial<Zm<M, C>> {
    UnivariatePolynomial::new(c.iter().map(|&a| Zm::new(a)).collect())
}

fn laurent<R: Ring>(terms: &[(i32, R)]) -> LaurentPolynomial<R> {
    let mut f = LaurentPolynomial::new();
    for (e, c) in terms {
        f.add_term(c.clone(), *e);
    }
    f
}

fn zm_laurent<const M: u64, const C: bool>(terms: &[(i32, i64)]) -> LaurentPolynomial<Zm<M, C>> {
    laurent(&terms.iter().map(|&(e, c)| (e, Zm::new(c))).collect::<Vec<_>>())
}

fn z_laurent(terms: &[(i32, i64)]) -> LaurentPolynomial<Integer> {
    laurent(&terms.iter().map(|&(e, c)| (e, Integer::from(c))).collect::<Vec<_>>())
}

/// `f·g == 1 == g·f`.
fn assert_two_sided<R: Ring>(f: &R, g: &R) {
    assert!((f.clone() * g.clone()).is_one(), "f·g != 1 for f = {f}, g = {g}");
    assert!((g.clone() * f.clone()).is_one(), "g·f != 1 for f = {f}, g = {g}");
}

fn not_implemented<T: fmt::Debug>(r: rustmath_core::Result<T>) -> String {
    match r {
        Err(MathError::NotImplemented(m)) => m,
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// UnivariatePolynomial.

#[test]
fn z_polynomial_units() {
    for f in [zx(&[1, 2]), zx(&[-1, 0, 2]), zx(&[0, 1]), zx(&[0]), zx(&[2])] {
        assert_eq!(f.unit_status(), Some(false), "{f}");
    }
    for f in [zx(&[-1]), zx(&[1])] {
        assert_eq!(f.unit_status(), Some(true), "{f}");
    }
    assert_eq!(zx(&[-1]).unit_inverse(), Ok(zx(&[-1])));
    assert_eq!(zx(&[1, 2]).unit_inverse(), Err(MathError::NotInvertible));
    assert_eq!(zx(&[-1]).multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(zx(&[0, 1]).multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
}

#[test]
fn z_polynomial_nilpotents_and_statics() {
    assert_eq!(zx(&[0, 2]).nilpotent_status(), Some(false));
    assert_eq!(zx(&[3, 1]).nilpotent_status(), Some(false));
    assert_eq!(zx(&[0]).nilpotent_status(), Some(true));
    assert_eq!(UnivariatePolynomial::<Integer>::static_characteristic(), Some(0));
    assert_eq!(UnivariatePolynomial::<Integer>::static_is_commutative(), Some(true));
    assert_eq!(UnivariatePolynomial::<i64>::static_characteristic(), Some(0));
    assert_eq!(UnivariatePolynomial::<i64>::static_is_commutative(), Some(true));
    let x64 = UnivariatePolynomial::new(vec![0i64, 2]);
    assert_eq!(x64.unit_status(), Some(false));
    assert_eq!(x64.nilpotent_status(), Some(false));
}

#[test]
fn q_polynomial_units() {
    let half = Rational::new(Integer::from(1), Integer::from(2)).unwrap();
    let two = UnivariatePolynomial::new(vec![Rational::from(2i64)]);
    assert_eq!(two.unit_status(), Some(true));
    assert_eq!(two.unit_inverse(), Ok(UnivariatePolynomial::new(vec![half])));
    let y1 = UnivariatePolynomial::new(vec![
        Rational::from(1i64),
        Rational::from(1i64),
    ]);
    assert_eq!(y1.unit_status(), Some(false));
    assert_eq!(UnivariatePolynomial::<Rational>::static_is_commutative(), Some(true));
}

#[test]
fn zmod8_units_and_inverse() {
    let f = zm::<8, true>(&[1, 2]);
    assert_eq!(f.unit_status(), Some(true));
    let h = f.unit_inverse().unwrap();
    assert_eq!(h, zm(&[1, 6, 4])); // 1 - 2t + 4t^2
    assert_two_sided(&f, &h);
    assert_eq!(f.multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    // An interior zero coefficient: the stop rule needs n = 2 consecutive zeros.
    let g = zm::<8, true>(&[1, 4, 2]);
    assert_eq!(g.unit_status(), Some(true));
    let gi = g.unit_inverse().unwrap();
    assert_eq!(gi, zm(&[1, 4, 6, 0, 4]));
    assert_two_sided(&g, &gi);
    assert_eq!(zm::<8, true>(&[2, 4]).nilpotent_status(), Some(true));
    let n = zm::<8, true>(&[1, 3]);
    assert_eq!(n.unit_status(), Some(false));
    assert_eq!(n.nilpotent_status(), Some(false));
    assert_eq!(n.unit_inverse(), Err(MathError::NotInvertible));
    assert_eq!(n.multiplicative_order_status(), MultiplicativeOrder::NotAUnit);
}

#[test]
fn zmod12_sage_examples() {
    let f = zm::<12, true>(&[5, 6]);
    assert_eq!(f.unit_status(), Some(true));
    assert_eq!(f.unit_inverse(), Ok(f.clone()));
    assert_eq!(f.nilpotent_status(), Some(false));
    assert_eq!(f.multiplicative_order_status(), MultiplicativeOrder::Finite(2));
    assert_eq!(zm::<12, true>(&[6, 0, 6]).nilpotent_status(), Some(true));
    assert_eq!(Z12::static_characteristic(), Some(12));
}

#[test]
fn commutativity_unknown_stays_undecided() {
    let f = zm::<8, false>(&[1, 2]);
    assert_eq!(f.unit_status(), None);
    let h = f.unit_inverse().unwrap();
    assert_eq!(h, zm(&[1, 6, 4]));
    assert_two_sided(&f, &h);
    // Not proven a unit, so the order search does not run.
    assert_eq!(f.multiplicative_order_status(), MultiplicativeOrder::Undecided);
    // A unit leading coefficient decides in every ring.
    assert_eq!(zm::<8, false>(&[1, 1]).unit_status(), Some(false));
    assert_eq!(UnivariatePolynomial::<Z8NoFlag>::static_is_commutative(), None);
}

#[test]
fn noncommutative_matrix_coefficients() {
    // I + A x + E11 x^2 with A = [[0,1],[1,0]] is the matrix [[1+x^2, x],[x, 1]] of
    // determinant 1; its inverse is I - A x + E22 x^2 although A is not nilpotent.
    let f = UnivariatePolynomial::new(vec![I2, SWAP, E11]);
    assert_eq!(f.unit_status(), None);
    let h = f.unit_inverse().unwrap();
    assert_eq!(h, UnivariatePolynomial::new(vec![I2, -SWAP, E22]));
    assert_two_sided(&f, &h);
    // I + E11 x has determinant x + 1, not a unit; the recurrence never stops.
    let g = UnivariatePolynomial::new(vec![I2, E11]);
    assert_eq!(g.unit_status(), None);
    assert!(not_implemented(g.unit_inverse()).contains("65536"));
    // A unit leading coefficient: not a unit in every ring.
    assert_eq!(UnivariatePolynomial::new(vec![I2, I2]).unit_status(), Some(false));
    // E12 (1 + x) is nilpotent, but the rule needs commutativity.
    assert_eq!(UnivariatePolynomial::new(vec![E12, E12]).nilpotent_status(), None);
}

#[test]
fn unit_inverse_work_bound() {
    // 1 + 2t over Z/2^62: h_k = (-2)^k, zero from k = 62 on, so the inverse has degree 61.
    let f = zm::<{ 1 << 62 }, true>(&[1, 2]);
    let h = f.unit_inverse().unwrap();
    assert_eq!(h.degree(), Some(61));
    let mut expected = Vec::new();
    let mut c = Z2e62::new(1);
    for _ in 0..62 {
        expected.push(c);
        c = c * Z2e62::new(-2);
    }
    assert_eq!(h, UnivariatePolynomial::new(expected));
    assert_two_sided(&f, &h);
    // 1 + 4t over Z/12 is not a unit (4 is not nilpotent), but without the commutativity
    // flag the recurrence (h_k = 8, 4, 8, ...) runs into the work bound.
    let g = zm::<12, false>(&[1, 4]);
    assert_eq!(g.unit_status(), None);
    let m = not_implemented(g.unit_inverse());
    assert!(m.contains("65536"), "{m}");
    // With the flag it is decided at once.
    let g = zm::<12, true>(&[1, 4]);
    assert_eq!(g.unit_status(), Some(false));
    assert_eq!(g.unit_inverse(), Err(MathError::NotInvertible));
}

// ---------------------------------------------------------------------------------------
// LaurentPolynomial.

#[test]
fn laurent_hooks() {
    // Over Z: monomials.
    let f = z_laurent(&[(5, -1)]);
    assert_eq!(f.unit_status(), Some(true));
    assert_eq!(f.unit_inverse(), Ok(z_laurent(&[(-5, -1)])));
    assert_eq!(f.multiplicative_order_status(), MultiplicativeOrder::Infinite);
    assert_eq!(z_laurent(&[(-2, 3)]).unit_status(), Some(false));
    assert_eq!(z_laurent(&[(-2, 3)]).unit_inverse(), Err(MathError::NotInvertible));
    // x + x^-1: both extreme coefficients are units, so no unit (every ring).
    assert_eq!(z_laurent(&[(1, 1), (-1, 1)]).unit_status(), Some(false));
    let top = z_laurent(&[(i32::MAX, 1)]);
    assert_eq!(top.unit_inverse(), Ok(z_laurent(&[(-i32::MAX, 1)])));
    let bottom = z_laurent(&[(i32::MIN, 1)]);
    assert_eq!(bottom.unit_status(), Some(true));
    assert_eq!(bottom.multiplicative_order_status(), MultiplicativeOrder::Infinite);
    assert!(not_implemented(bottom.unit_inverse()).contains("no negative in i32"));
    assert_eq!(LaurentPolynomial::<Integer>::static_characteristic(), Some(0));
    assert_eq!(LaurentPolynomial::<Integer>::static_is_commutative(), Some(true));

    // Over Z/8: 1 + 2x^-1 (high side), x^-3 + 2x^-2 (low side).
    let g = zm_laurent::<8, true>(&[(0, 1), (-1, 2)]);
    assert_eq!(g.unit_status(), Some(true));
    let gi = g.unit_inverse().unwrap();
    assert_eq!(gi, zm_laurent(&[(0, 1), (-1, 6), (-2, 4)]));
    assert_two_sided(&g, &gi);
    assert_eq!(g.multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    let k = zm_laurent::<8, true>(&[(-3, 1), (-2, 2)]);
    assert_eq!(k.unit_status(), Some(true));
    let ki = k.unit_inverse().unwrap();
    assert_eq!(ki, zm_laurent(&[(3, 1), (4, 6), (5, 4)]));
    assert_two_sided(&k, &ki);
    assert_eq!(zm_laurent::<8, true>(&[(3, 2)]).nilpotent_status(), Some(true));
    // Commutative: nilpotent iff every coefficient is (2x^-1 + 4x over Z/8).
    assert_eq!(zm_laurent::<8, true>(&[(-1, 2), (1, 4)]).nilpotent_status(), Some(true));
    // ... and a non-nilpotent interior coefficient decides "not nilpotent".
    assert_eq!(
        zm_laurent::<8, true>(&[(-1, 2), (0, 1), (1, 4)]).nilpotent_status(),
        Some(false)
    );

    // Over Z/6: 3 + 4x is a unit with inverse 3 + 4x^-1, but no extreme coefficient is a
    // unit: undecided (documented limit).
    let z6 = zm_laurent::<6, true>(&[(0, 3), (1, 4)]);
    assert_eq!(z6.unit_status(), None);
    not_implemented(z6.unit_inverse());
    let z6_inv = zm_laurent::<6, true>(&[(0, 3), (-1, 4)]);
    assert!((z6.clone() * z6_inv).is_one());

    // Every ring: two unit extreme coefficients (I x^-1 + I x over 2×2 matrices).
    let m = laurent(&[(-1, I2), (1, I2)]);
    assert_eq!(m.unit_status(), Some(false));
}

/// Z/8 whose static constructors panic, like a parent-carrying coefficient type without a
/// sentinel: the polynomial hooks must read only the coefficients.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NoStatic(Zm<8, true>);

impl fmt::Display for NoStatic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for NoStatic {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        NoStatic(self.0 + o.0)
    }
}
impl Sub for NoStatic {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        NoStatic(self.0 - o.0)
    }
}
impl Mul for NoStatic {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        NoStatic(self.0 * o.0)
    }
}
impl Neg for NoStatic {
    type Output = Self;
    fn neg(self) -> Self {
        NoStatic(-self.0)
    }
}
impl Ring for NoStatic {
    fn zero() -> Self {
        panic!("NoStatic::zero called")
    }
    fn one() -> Self {
        panic!("NoStatic::one called")
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
    fn unit_status(&self) -> Option<bool> {
        self.0.unit_status()
    }
    fn nilpotent_status(&self) -> Option<bool> {
        self.0.nilpotent_status()
    }
    fn unit_inverse(&self) -> rustmath_core::Result<Self> {
        self.0.unit_inverse().map(NoStatic)
    }
    fn static_is_commutative() -> Option<bool> {
        Some(true)
    }
}

#[test]
fn hooks_call_no_static_constructor() {
    let c = |a: i64| NoStatic(Zm::new(a));
    // 1 + 4t + 2t^2 over Z/8: the inverse 1 + 4t + 6t^2 + 4t^4 has an interior zero.
    let f = UnivariatePolynomial::new(vec![c(1), c(4), c(2)]);
    assert_eq!(f.unit_status(), Some(true));
    assert_eq!(f.nilpotent_status(), Some(false));
    let h = f.unit_inverse().unwrap();
    assert_eq!(h.coefficients(), &[c(1), c(4), c(6), c(0), c(4)]);
    let g = UnivariatePolynomial::new(vec![c(1), c(2)]);
    assert_eq!(g.multiplicative_order_status(), MultiplicativeOrder::Finite(4));
    // Laurent: x^-3 + 2x^-2 has a gap-free span; x^-3 + 2x^-1 has a gap filled by c - c.
    let mut l = LaurentPolynomial::new();
    l.add_term(c(1), -3);
    l.add_term(c(2), -1);
    assert_eq!(l.unit_status(), Some(true));
    let li = l.unit_inverse().unwrap();
    // x^-3 (1 + 2x^2): inverse x^3 (1 + 6x^2 + 4x^4).
    assert_eq!(li.terms(), vec![(3, c(1)), (5, c(6)), (7, c(4))]);
    // f^k = x^(-3k) (1 + 2x^2)^k is never 1, but the bounded search cannot prove that.
    assert_eq!(l.multiplicative_order_status(), MultiplicativeOrder::Undecided);
}
