//! MAGMA Handbook ch. 23 (Univariate Polynomial Rings), digest
//! `rustmath-lava/docs/Chapter_23_Univariate_Polynomial_Rings.md`:
//! §23.4.10 `Quotrem`, `div`, `mod`, `IsDivisibleBy`, `ExactQuotient`, `PseudoRemainder` and
//! §23.5 `GCD`, `LCM`, `Normalize`, `Content`, `PrimitivePart`, `ContentAndPrimitivePart`
//! for `UnivariatePolynomial` over Z, Q and F_p.
//!
//! The digest has no worked example (H23E*) for these sections, so the reference values come
//! from PARI/GP (`gp -q`); each test names the command. PARI's `divrem` returns `[q, r]`;
//! over Q and F_p PARI does not make gcd/lcm monic, so those values are compared after
//! `mon(p) = p/pollead(p)`. Over Z, PARI's gcd/lcm are content-aware with positive leading
//! coefficient, i.e. already MAGMA-normalized, except that `gcd(0, p)` returns `p` as is.
//!
//! Inputs marked "formerly hung" looped forever before the fix; they run inside `within`, so
//! a regression fails fast instead of stalling the test run.

use rustmath_core::{
    CommutativeRing, EuclideanDomain, Field, IntegralDomain, MathError, Result, Ring,
};
use rustmath_integers::Integer;
use rustmath_polynomials::{zx, UnivariatePolynomial};
use rustmath_rationals::Rational;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::mpsc;
use std::time::Duration;

type ZPoly = UnivariatePolynomial<Integer>;
type QPoly = UnivariatePolynomial<Rational>;

/// Little-endian integer polynomial.
fn z(c: &[i64]) -> ZPoly {
    UnivariatePolynomial::new(c.iter().map(|&a| Integer::from(a)).collect())
}

fn rat(n: i64, d: i64) -> Rational {
    Rational::new(Integer::from(n), Integer::from(d)).unwrap()
}

/// Little-endian rational polynomial from (numerator, denominator) pairs.
fn q(c: &[(i64, i64)]) -> QPoly {
    UnivariatePolynomial::new(c.iter().map(|&(n, d)| rat(n, d)).collect())
}

/// Little-endian rational polynomial with integer coefficients.
fn qi(c: &[i64]) -> QPoly {
    UnivariatePolynomial::new(c.iter().map(|&a| rat(a, 1)).collect())
}

fn z_to_q(p: &ZPoly) -> QPoly {
    UnivariatePolynomial::new(
        p.coefficients()
            .iter()
            .map(|a| Rational::from_integer(a.clone()))
            .collect(),
    )
}

/// Run `f` on a thread and fail if it does not finish within `secs` seconds.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .expect("timed out: the old non-terminating division is back")
}

// ---------------------------------------------------------------------------------------
// A test-local prime field F_P (GFp in rustmath-rings cannot be used: rings depends on
// polynomials). div_rem(a, b) = (a·b⁻¹, 0), Err(DivisionByZero) for b = 0.
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fp<const P: u64>(u64);

impl<const P: u64> Fp<P> {
    fn new(a: i64) -> Self {
        Fp(a.rem_euclid(P as i64) as u64)
    }
    fn inv(self) -> Option<Self> {
        if self.0 == 0 {
            return None;
        }
        // Fermat: a^(P-2)
        let (mut b, mut e, mut acc) = (self.0, P - 2, 1u64);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * b % P;
            }
            b = b * b % P;
            e >>= 1;
        }
        Some(Fp(acc))
    }
}
impl<const P: u64> fmt::Display for Fp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl<const P: u64> Add for Fp<P> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Fp((self.0 + o.0) % P)
    }
}
impl<const P: u64> Sub for Fp<P> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Fp((self.0 + P - o.0) % P)
    }
}
impl<const P: u64> Mul for Fp<P> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Fp(self.0 * o.0 % P)
    }
}
impl<const P: u64> Neg for Fp<P> {
    type Output = Self;
    fn neg(self) -> Self {
        Fp((P - self.0) % P)
    }
}
impl<const P: u64> Div for Fp<P> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        self * o.inv().expect("division by zero in F_p")
    }
}
impl<const P: u64> Ring for Fp<P> {
    fn zero() -> Self {
        Fp(0)
    }
    fn one() -> Self {
        Fp(1 % P)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
    fn is_one(&self) -> bool {
        self.0 == 1
    }
}
impl<const P: u64> CommutativeRing for Fp<P> {}
impl<const P: u64> IntegralDomain for Fp<P> {}
impl<const P: u64> EuclideanDomain for Fp<P> {
    fn norm(&self) -> u64 {
        u64::from(self.0 != 0)
    }
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        match other.inv() {
            Some(i) => Ok((*self * i, Fp(0))),
            None => Err(MathError::DivisionByZero),
        }
    }
}
impl<const P: u64> Field for Fp<P> {
    fn inverse(&self) -> Result<Self> {
        self.inv().ok_or(MathError::DivisionByZero)
    }
}

fn fp<const P: u64>(c: &[i64]) -> UnivariatePolynomial<Fp<P>> {
    UnivariatePolynomial::new(c.iter().map(|&a| Fp::<P>::new(a)).collect())
}

/// Deterministic 64-bit LCG (Knuth MMIX constants).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
    fn zpoly(&mut self, max_deg: i64, bound: i64) -> ZPoly {
        let d = self.range(0, max_deg) as usize;
        z(&(0..=d)
            .map(|_| self.range(-bound, bound))
            .collect::<Vec<_>>())
    }
    fn nonzero_zpoly(&mut self, max_deg: i64, bound: i64) -> ZPoly {
        loop {
            let p = self.zpoly(max_deg, bound);
            if !p.is_zero() {
                return p;
            }
        }
    }
}

fn lc(p: &ZPoly) -> Integer {
    p.leading_coefficient().unwrap().clone()
}

// ---------------------------------------------------------------------------------------
// §23.4.10 Quotrem over Q (classical division: deg r < deg g, unique)
// ---------------------------------------------------------------------------------------

// gp: divrem(x^3+2*x+1, 2*x^2+1)            -> [1/2*x, 3/2*x + 1]
//     divrem(3*x^4-2*x^3+x-7, 2*x^2+3*x-1)  -> [3/2*x^2 - 13/4*x + 45/8, -153/8*x - 11/8]
//     divrem(x^5-1, x^2+x+1)                -> [x^3 - x^2 + 1, -x - 2]
//     divrem(6*x^3+x^2+1, 2*x+1)            -> [3*x^2 - x + 1/2, 1/2]
//     divrem(x^2, 2*x)                      -> [1/2*x, 0]
//     divrem(3*x^2+x, 2*x)                  -> [3/2*x + 1/2, 0]
#[test]
fn quotrem_over_q_matches_pari() {
    let cases: Vec<(QPoly, QPoly, QPoly, QPoly)> = vec![
        (
            qi(&[1, 2, 0, 1]),
            qi(&[1, 0, 2]),
            q(&[(0, 1), (1, 2)]),
            q(&[(1, 1), (3, 2)]),
        ),
        (
            qi(&[-7, 1, 0, -2, 3]),
            qi(&[-1, 3, 2]),
            q(&[(45, 8), (-13, 4), (3, 2)]),
            q(&[(-11, 8), (-153, 8)]),
        ),
        (
            qi(&[-1, 0, 0, 0, 0, 1]),
            qi(&[1, 1, 1]),
            qi(&[1, 0, -1, 1]),
            qi(&[-2, -1]),
        ),
        (
            qi(&[1, 0, 1, 6]),
            qi(&[1, 2]),
            q(&[(1, 2), (-1, 1), (3, 1)]),
            q(&[(1, 2)]),
        ),
        (qi(&[0, 0, 1]), qi(&[0, 2]), q(&[(0, 1), (1, 2)]), qi(&[0])),
        (qi(&[0, 1, 3]), qi(&[0, 2]), q(&[(1, 2), (3, 2)]), qi(&[0])),
    ];
    for (f, g, eq, er) in cases {
        let (qq, rr) = f.div_rem(&g).unwrap();
        assert_eq!((&qq, &rr), (&eq, &er), "divrem({f}, {g})");
        assert_eq!(qq.clone() * g.clone() + rr.clone(), f);
        assert_eq!(f.clone() / g.clone(), eq);
        assert_eq!(f.clone() % g.clone(), er);
        assert_eq!(f.quo_rem(&g), (eq, er));
    }
}

// ---------------------------------------------------------------------------------------
// §23.4.10 Quotrem over Z: deg r minimal; the division stops at the first leading
// coefficient that lc(g) does not divide. Where every step is exact the result is the
// Z-image of PARI's divrem; otherwise the identity f = q·g + r is checked with gp.
// ---------------------------------------------------------------------------------------

#[test]
fn quotrem_over_z_terminates_with_minimal_degree() {
    let cases: Vec<(ZPoly, ZPoly, ZPoly, ZPoly, &str)> = vec![
        // formerly hung: 2 does not divide 1 at the first step
        (
            z(&[0, 0, 1]),
            z(&[0, 2]),
            z(&[0]),
            z(&[0, 0, 1]),
            "x^2 by 2x",
        ),
        // formerly hung
        (
            z(&[0, 1, 3]),
            z(&[0, 2]),
            z(&[0]),
            z(&[0, 1, 3]),
            "3x^2+x by 2x",
        ),
        // negative truncation path: -3 = (-1)·2 + (-1)
        (
            z(&[0, 0, -3]),
            z(&[0, 2]),
            z(&[0]),
            z(&[0, 0, -3]),
            "-3x^2 by 2x",
        ),
        // two exact steps, then 2 does not divide 1.
        // gp: (3*x^2-x)*(2*x+1)+(x+1) -> 6*x^3 + x^2 + 1 (deg r = deg g = 1, minimal)
        (
            z(&[1, 0, 1, 6]),
            z(&[1, 2]),
            z(&[0, -1, 3]),
            z(&[1, 1]),
            "6x^3+x^2+1 by 2x+1",
        ),
        // gp: divrem(4*x^3+6*x^2+3, 2*x+1) -> [2*x^2 + 2*x - 1, 4] (all steps exact)
        (
            z(&[3, 0, 6, 4]),
            z(&[1, 2]),
            z(&[-1, 2, 2]),
            z(&[4]),
            "4x^3+6x^2+3 by 2x+1",
        ),
        // gp: divrem(x^4+3*x+5, x^2-2) -> [x^2 + 2, 3*x + 9]
        (
            z(&[5, 3, 0, 0, 1]),
            z(&[-2, 0, 1]),
            z(&[2, 0, 1]),
            z(&[9, 3]),
            "x^4+3x+5 by x^2-2",
        ),
        // gp: divrem(3*x^2+1, Pol(-1)) -> [-3*x^2 - 1, 0]
        (
            z(&[1, 0, 3]),
            z(&[-1]),
            z(&[-1, 0, -3]),
            z(&[0]),
            "3x^2+1 by -1",
        ),
        // non-unit constant divisor: 2 does not divide 3
        (
            z(&[1, 0, 3]),
            z(&[2]),
            z(&[0]),
            z(&[1, 0, 3]),
            "3x^2+1 by 2",
        ),
        // zero dividend, smaller degree
        (z(&[0]), z(&[0, 2]), z(&[0]), z(&[0]), "0 by 2x"),
        (
            z(&[1, 1]),
            z(&[0, 0, 2]),
            z(&[0]),
            z(&[1, 1]),
            "x+1 by 2x^2",
        ),
    ];
    for (f, g, eq, er, name) in cases {
        let (f2, g2) = (f.clone(), g.clone());
        let (qq, rr) = within(5, move || f2.div_rem(&g2).unwrap());
        assert_eq!((&qq, &rr), (&eq, &er), "{name}");
        assert_eq!(qq * g.clone() + rr, f, "{name}: f = q·g + r");
        assert_eq!(f.quo_rem(&g), (eq, er), "{name}: quo_rem");
    }
}

#[test]
fn div_and_mod_operators_over_z_formerly_hung() {
    let (qq, rr) = within(5, || {
        (z(&[0, 0, 1]) / z(&[0, 2]), z(&[0, 0, 1]) % z(&[0, 2]))
    });
    assert_eq!(qq, z(&[0]));
    assert_eq!(rr, z(&[0, 0, 1]));
}

#[test]
fn division_by_zero_is_an_error() {
    assert_eq!(z(&[1, 2]).div_rem(&z(&[0])), Err(MathError::DivisionByZero));
    assert_eq!(
        qi(&[1, 2]).div_rem(&qi(&[0])),
        Err(MathError::DivisionByZero)
    );
    assert_eq!(z(&[0]).div_rem(&z(&[0])), Err(MathError::DivisionByZero));
}

#[test]
#[should_panic(expected = "Division by zero polynomial")]
fn quo_rem_by_zero_panics() {
    let _ = z(&[1, 2]).quo_rem(&z(&[0]));
}

#[test]
#[should_panic(expected = "polynomial div/mod")]
fn div_operator_by_zero_panics() {
    let _ = z(&[1, 2]) / z(&[0]);
}

/// Invariants of MAGMA `Quotrem` on 300 pseudo-random pairs over Z.
#[test]
fn quotrem_over_z_invariants() {
    let mut rng = Lcg(0x5eed_0023);
    for _ in 0..300 {
        let f = rng.zpoly(6, 9);
        let g = rng.nonzero_zpoly(6, 9);
        let (qq, rr) = f.div_rem(&g).unwrap();
        assert_eq!(
            qq.clone() * g.clone() + rr.clone(),
            f,
            "f = q·g + r for f = {f}, g = {g}"
        );
        let lg = lc(&g);
        if lg == Integer::from(1) || lg == Integer::from(-1) {
            assert!(
                rr.degree() < g.degree(),
                "unit lc: deg r < deg g for f = {f}, g = {g}"
            );
        }
        if let (Some(dr), Some(dg)) = (rr.degree(), g.degree()) {
            if dr >= dg {
                // stopped at a leading coefficient lc(g) does not divide
                let (_, rho) = lc(&rr).div_rem(&lg).unwrap();
                assert!(!rho.is_zero(), "minimality for f = {f}, g = {g}");
            }
        }
        assert_eq!(rr.is_zero(), f.exact_quotient(&g).is_ok());
        assert_eq!(
            (f.clone() * g.clone()).div_rem(&g),
            Ok((f.clone(), z(&[0])))
        );
    }
}

// F_7: f7 = Mod(1,7)*((x+1)^2*(x+3)) = x^3 + 5x^2 + 3, g7 = Mod(1,7)*(3*(x+1)*(x+5)) = 3x^2 + 4x + 1.
// gp: divrem(f7, g7) -> [Mod(5,7)*x + Mod(2,7), Mod(1,7)*x + Mod(1,7)]
//     divrem(x^2*Mod(1,7), 2*x*Mod(1,7)) -> [Mod(4,7)*x, Mod(0,7)]
#[test]
fn quotrem_over_f7_matches_pari() {
    let f7 = fp::<7>(&[3, 0, 5, 1]);
    let g7 = fp::<7>(&[1, 4, 3]);
    assert_eq!(
        f7.div_rem(&g7).unwrap(),
        (fp::<7>(&[2, 5]), fp::<7>(&[1, 1]))
    );
    assert_eq!(
        fp::<7>(&[0, 0, 1]).div_rem(&fp::<7>(&[0, 2])).unwrap(),
        (fp::<7>(&[0, 4]), fp::<7>(&[0]))
    );
}

// ---------------------------------------------------------------------------------------
// §23.4.10 ExactQuotient / IsDivisibleBy
// ---------------------------------------------------------------------------------------

// gp: (2*x^2+2*x)/(2*x) -> x + 1;  (x^3-1)/(x-1) -> x^2 + x + 1;
//     (x^2)/(2*x) -> 1/2*x (not integral, so 2x does not divide x^2 in Z[x]);
//     (x^2-1/4)/(x+1/2) -> x - 1/2
#[test]
fn exact_quotient_matches_pari() {
    assert_eq!(z(&[0, 2, 2]).exact_quotient(&z(&[0, 2])), Ok(z(&[1, 1])));
    assert_eq!(
        z(&[-1, 0, 0, 1]).exact_quotient(&z(&[-1, 1])),
        Ok(z(&[1, 1, 1]))
    );
    assert!(matches!(
        z(&[0, 0, 1]).exact_quotient(&z(&[0, 2])),
        Err(MathError::NotDivisible(_))
    ));
    assert_eq!(
        q(&[(-1, 4), (0, 1), (1, 1)]).exact_quotient(&q(&[(1, 2), (1, 1)])),
        Ok(q(&[(-1, 2), (1, 1)]))
    );
    assert_eq!(
        z(&[1, 1]).exact_quotient(&z(&[0])),
        Err(MathError::DivisionByZero)
    );
    assert_eq!(z(&[0]).exact_quotient(&z(&[0, 2])), Ok(z(&[0])));
}

#[test]
fn is_divisible_by_matches_pari() {
    assert_eq!(
        z(&[0, 2, 2]).is_divisible_by(&z(&[0, 2])),
        Ok(Some(z(&[1, 1])))
    );
    assert_eq!(z(&[0, 0, 1]).is_divisible_by(&z(&[0, 2])), Ok(None));
    assert_eq!(
        z(&[1, 1]).is_divisible_by(&z(&[0])),
        Err(MathError::DivisionByZero)
    );
    // 0 = q·0 for every q: no quotient to return, so this is an error, never "not divisible".
    assert_eq!(
        z(&[0]).is_divisible_by(&z(&[0])),
        Err(MathError::DivisionByZero)
    );
    assert_eq!(z(&[0]).is_divisible_by(&z(&[0, 2])), Ok(Some(z(&[0]))));
    assert_eq!(
        qi(&[0, 0, 1]).is_divisible_by(&qi(&[0, 2])),
        Ok(Some(q(&[(0, 1), (1, 2)])))
    );
}

// ---------------------------------------------------------------------------------------
// §23.4.10 PseudoRemainder: lc(g)^d·f = q·g + r, d = max(0, deg f - deg g + 1)
// gp: prq(f,g) = my(d=max(0,poldegree(f)-poldegree(g)+1), c=pollead(g)); [(c^d*f)\g, (c^d*f)%g]
// ---------------------------------------------------------------------------------------

#[test]
fn pseudo_quo_rem_matches_pari() {
    let cases: Vec<(ZPoly, ZPoly, ZPoly, ZPoly)> = vec![
        // prq(x^2, 2*x) -> [2*x, 0]
        (z(&[0, 0, 1]), z(&[0, 2]), z(&[0, 2]), z(&[0])),
        // prq(6*x^3+x^2+1, 2*x+1) -> [24*x^2 - 8*x + 4, 4]
        (z(&[1, 0, 1, 6]), z(&[1, 2]), z(&[4, -8, 24]), z(&[4])),
        // prq(3*x^4+x+7, 2*x^2+1) -> [12*x^2 - 6, 8*x + 62]: one step drops the degree by 2,
        // which pins the full exponent d = 3
        (
            z(&[7, 1, 0, 0, 3]),
            z(&[1, 0, 2]),
            z(&[-6, 0, 12]),
            z(&[62, 8]),
        ),
        // Knuth TAOCP §4.6.1 example:
        // prq(x^8+x^6-3*x^4-3*x^3+8*x^2+2*x-5, 3*x^6+5*x^4-4*x^2-9*x+21) -> [9*x^2 - 6, -15*x^4 + 3*x^2 - 9]
        (
            z(&[-5, 2, 8, -3, -3, 0, 1, 0, 1]),
            z(&[21, -9, -4, 0, 5, 0, 3]),
            z(&[-6, 0, 9]),
            z(&[-9, 0, 3, 0, -15]),
        ),
        // prq(x+1, x^2) -> [0, x + 1] (d = 0)
        (z(&[1, 1]), z(&[0, 0, 1]), z(&[0]), z(&[1, 1])),
        // prq(5*x^3-x+2, -3*x^2+x) -> [-15*x - 5, -4*x + 18] (negative lc)
        (
            z(&[2, -1, 0, 5]),
            z(&[0, 1, -3]),
            z(&[-5, -15]),
            z(&[18, -4]),
        ),
        // prq(3*x^2+1, Pol(2)) -> [12*x^2 + 4, 0] (constant divisor, d = 3)
        (z(&[1, 0, 3]), z(&[2]), z(&[4, 0, 12]), z(&[0])),
        // zero dividend
        (z(&[0]), z(&[1, 2]), z(&[0]), z(&[0])),
    ];
    for (f, g, eq, er) in cases {
        let (qq, rr) = f.pseudo_quo_rem(&g).unwrap();
        assert_eq!((&qq, &rr), (&eq, &er), "prq({f}, {g})");
        assert_eq!(f.pseudo_remainder(&g).unwrap(), er);
        let d = match (f.degree(), g.degree()) {
            (Some(df), Some(dg)) if df + 1 >= dg => (df + 1 - dg) as u32,
            _ => 0,
        };
        let c = lc(&g).pow(d);
        assert_eq!(f.scalar_mul(&c), qq * g.clone() + rr, "lc(g)^d·f = q·g + r");
    }
    assert_eq!(
        z(&[1, 2]).pseudo_quo_rem(&z(&[0])),
        Err(MathError::DivisionByZero)
    );
    assert_eq!(
        z(&[1, 2]).pseudo_remainder(&z(&[0])),
        Err(MathError::DivisionByZero)
    );
}

// ---------------------------------------------------------------------------------------
// §23.5.1 GCD / LCM over Z: content-aware, positive leading coefficient (gp -q, PARI gcd/lcm)
// ---------------------------------------------------------------------------------------

#[test]
fn gcd_lcm_over_z_match_pari() {
    // (f, g, gcd, lcm); None = lcm not checked here
    let cases: Vec<(ZPoly, ZPoly, ZPoly, Option<ZPoly>)> = vec![
        // gcd(x^2, 2*x) -> x; lcm(x^2, 2*x) -> 2*x^2 (formerly hung)
        (z(&[0, 0, 1]), z(&[0, 2]), z(&[0, 1]), Some(z(&[0, 0, 2]))),
        // gcd(2*x^2-2, 4*x+4) -> 2*x + 2; lcm -> 4*x^2 - 4 (formerly hung)
        (z(&[-2, 0, 2]), z(&[4, 4]), z(&[2, 2]), Some(z(&[-4, 0, 4]))),
        // gcd(-x^3+4*x^2-5*x+2, -3*x^2+8*x-5) -> x - 1 (formerly hung)
        (z(&[2, -5, 4, -1]), z(&[-5, 8, -3]), z(&[-1, 1]), None),
        // gcd(6*x^2+12*x+6, 4*x^2-4) -> 2*x + 2; lcm -> 12*x^3 + 12*x^2 - 12*x - 12
        (
            z(&[6, 12, 6]),
            z(&[-4, 0, 4]),
            z(&[2, 2]),
            Some(z(&[-12, -12, 12, 12])),
        ),
        // gcd(-2*x-2, -4*x-4) -> 2*x + 2
        (z(&[-2, -2]), z(&[-4, -4]), z(&[2, 2]), None),
        // gcd(x+1, -x-1) -> x + 1
        (z(&[1, 1]), z(&[-1, -1]), z(&[1, 1]), None),
        // gcd(0*x, 0*x) -> 0
        (z(&[0]), z(&[0]), z(&[0]), Some(z(&[0]))),
        // gcd(4*x+4, 0) -> 4*x + 4; lcm(0, x) -> 0
        (z(&[4, 4]), z(&[0]), z(&[4, 4]), Some(z(&[0]))),
        // gcd(Pol(6), Pol(4)) -> 2; lcm(Pol(6), Pol(4)) -> 12
        (z(&[6]), z(&[4]), z(&[2]), Some(z(&[12]))),
        // gcd(2*x, 3) -> 1
        (z(&[0, 2]), z(&[3]), z(&[1]), None),
        // gcd(knuth u, v) -> 1
        (
            z(&[-5, 2, 8, -3, -3, 0, 1, 0, 1]),
            z(&[21, -9, -4, 0, 5, 0, 3]),
            z(&[1]),
            None,
        ),
        // bench_poly_lcm inputs: gcd(x^3+6*x^2+11*x+6, 3*x^2+5*x+2) -> x + 1;
        // lcm -> 3*x^4 + 20*x^3 + 45*x^2 + 40*x + 12 (formerly hung)
        (
            z(&[6, 11, 6, 1]),
            z(&[2, 5, 3]),
            z(&[1, 1]),
            Some(z(&[12, 40, 45, 20, 3])),
        ),
        // gcd((x-1)(x-2)(x-3), (x-2)(x-3)(x-4)) -> x^2 - 5*x + 6
        (
            z(&[-6, 11, -6, 1]),
            z(&[-24, 26, -9, 1]),
            z(&[6, -5, 1]),
            None,
        ),
        // gcd(x^2+x, 2*x+1) -> 1 (formerly hung)
        (z(&[0, 1, 1]), z(&[1, 2]), z(&[1]), None),
        // gcd(x^3-x^2-x+1, 3*x^2-2*x-1) -> x - 1
        (z(&[1, -1, -1, 1]), z(&[-1, -2, 3]), z(&[-1, 1]), None),
    ];
    for (f, g, eg, el) in cases {
        let (f2, g2) = (f.clone(), g.clone());
        let got = within(10, move || f2.gcd(&g2));
        assert_eq!(got, eg, "gcd({f}, {g})");
        assert_eq!(g.gcd(&f), eg, "gcd({g}, {f})");
        if let Some(el) = el {
            let (f2, g2) = (f.clone(), g.clone());
            let got = within(10, move || f2.lcm(&g2));
            assert_eq!(got, el, "lcm({f}, {g})");
        }
    }
}

// A = 12*(3*x^3-2*x+7)^2*(5*x^2+x-1), B = 18*(3*x^3-2*x+7)*(x^4+1)
//   gp: gcd(A,B) -> 18*x^3 - 12*x + 42
//       lcm(A,B) -> 1620*x^12 + 324*x^11 - 2484*x^10 + 7128*x^9 + 4284*x^8 - 6084*x^7
//                   + 5184*x^6 + 9900*x^5 + 900*x^4 - 6408*x^3 + 7668*x^2 + 2772*x - 1764
// C = 4*(x^2+x+1)*(x^5-3*x+1), D = 6*(x^2+x+1)*(2*x^2-7)   (delta = 3 in the subresultant update)
//   gp: gcd(C,D) -> 2*x^2 + 2*x + 2
//       lcm(C,D) -> 24*x^9 + 24*x^8 - 60*x^7 - 84*x^6 - 156*x^5 - 48*x^4 + 204*x^3 + 192*x^2 + 168*x - 84
// E = (6*x^2-3)*(2*x+5)*(7*x-1), F = (6*x^2-3)*(4*x^3+x+9)  (gcd with non-unit lc after content removal)
//   gp: gcd(E,F) -> 6*x^2 - 3
//       lcm(E,F) -> 336*x^7 + 792*x^6 - 204*x^5 + 558*x^4 + 1770*x^3 - 747*x^2 - 876*x + 135
#[test]
fn gcd_lcm_over_z_with_content_match_pari() {
    let a = z(&[12]) * z(&[7, -2, 0, 3]) * z(&[7, -2, 0, 3]) * z(&[-1, 1, 5]);
    let b = z(&[18]) * z(&[7, -2, 0, 3]) * z(&[1, 0, 0, 0, 1]);
    // gp: A -> 540*x^8 + 108*x^7 - 828*x^6 + 2376*x^5 + 888*x^4 - 2136*x^3 + 2556*x^2 + 924*x - 588
    assert_eq!(a, z(&[-588, 924, 2556, -2136, 888, 2376, -828, 108, 540]));
    assert_eq!(a.gcd(&b), z(&[42, -12, 0, 18]));
    assert_eq!(
        a.lcm(&b),
        z(&[-1764, 2772, 7668, -6408, 900, 9900, 5184, -6084, 4284, 7128, -2484, 324, 1620])
    );

    let c = z(&[4]) * z(&[1, 1, 1]) * z(&[1, -3, 0, 0, 0, 1]);
    let d = z(&[6]) * z(&[1, 1, 1]) * z(&[-7, 0, 2]);
    // gp: C -> 4*x^7 + 4*x^6 + 4*x^5 - 12*x^3 - 8*x^2 - 8*x + 4; D -> 12*x^4 + 12*x^3 - 30*x^2 - 42*x - 42
    assert_eq!(c, z(&[4, -8, -8, -12, 0, 4, 4, 4]));
    assert_eq!(d, z(&[-42, -42, -30, 12, 12]));
    assert_eq!(c.gcd(&d), z(&[2, 2, 2]));
    assert_eq!(
        c.lcm(&d),
        z(&[-84, 168, 192, 204, -48, -156, -84, -60, 24, 24])
    );

    let e = z(&[-3, 0, 6]) * z(&[5, 2]) * z(&[-1, 7]);
    let f = z(&[-3, 0, 6]) * z(&[9, 1, 0, 4]);
    // gp: E -> 84*x^4 + 198*x^3 - 72*x^2 - 99*x + 15; F -> 24*x^5 - 6*x^3 + 54*x^2 - 3*x - 27
    assert_eq!(e, z(&[15, -99, -72, 198, 84]));
    assert_eq!(f, z(&[-27, -3, 54, -6, 0, 24]));
    assert_eq!(e.gcd(&f), z(&[-3, 0, 6]));
    assert_eq!(e.lcm(&f), z(&[135, -876, -747, 1770, 558, -204, 792, 336]));
}

// MAGMA: "Result is normalized (unique). Zero inputs: GCD is the other polynomial".
// gp: gcd(0, -3*x-6) -> -3*x - 6 (PARI does not normalize when one input is 0), so the
// expected value is its normalization 3x + 6.
#[test]
fn gcd_with_zero_is_the_normalized_other_input() {
    assert_eq!(z(&[0]).gcd(&z(&[-6, -3])), z(&[6, 3]));
    assert_eq!(z(&[-6, -3]).gcd(&z(&[0])), z(&[6, 3]));
    assert_eq!(z(&[0]).lcm(&z(&[-6, -3])), z(&[0]));
}

// gp: gcd(x^2, 2*x) -> x, lcm(x^2, 2*x) -> 2*x^2 (through the i64 sign override)
#[test]
fn gcd_lcm_over_i64() {
    let x2 = UnivariatePolynomial::new(vec![0i64, 0, 1]);
    let tx = UnivariatePolynomial::new(vec![0i64, 2]);
    assert_eq!(x2.gcd(&tx), UnivariatePolynomial::new(vec![0i64, 1]));
    assert_eq!(x2.lcm(&tx), UnivariatePolynomial::new(vec![0i64, 0, 2]));
    let a = UnivariatePolynomial::new(vec![-2i64, 0, 2]);
    let b = UnivariatePolynomial::new(vec![-4i64, -4]);
    // gp: gcd(2*x^2-2, -4*x-4) -> 2*x + 2
    assert_eq!(a.gcd(&b), UnivariatePolynomial::new(vec![2i64, 2]));
}

/// Differential test against the independent slice implementation `zx::subresultant_gcd`
/// (Cohen Alg. 3.3.1, PARI-tested in `zx.rs`), plus the gcd/lcm identities, on 300
/// pseudo-random pairs of nonzero polynomials with a common factor.
#[test]
fn gcd_over_z_agrees_with_zx_and_satisfies_identities() {
    let mut rng = Lcg(0x0023_0005_0001);
    for _ in 0..300 {
        let common = rng.nonzero_zpoly(3, 6);
        let f = rng.nonzero_zpoly(5, 20) * common.clone();
        let g = rng.nonzero_zpoly(5, 20) * common.clone();
        let gg = f.gcd(&g);
        let reference = zx::trim(&zx::subresultant_gcd(f.coefficients(), g.coefficients()));
        assert_eq!(gg.coefficients(), &reference[..], "gcd({f}, {g}) vs zx");
        assert_eq!(gg, g.gcd(&f));
        assert!(lc(&gg) > Integer::from(0));
        assert_eq!(
            f.is_divisible_by(&gg).map(|o| o.is_some()),
            Ok(true),
            "gcd | f"
        );
        assert_eq!(
            g.is_divisible_by(&gg).map(|o| o.is_some()),
            Ok(true),
            "gcd | g"
        );
        let l = f.lcm(&g);
        assert_eq!(
            l.clone() * gg.clone(),
            (f.clone() * g.clone()).normalize(),
            "lcm·gcd = f·g"
        );
        // gcd(f·h, g·h) = normalize(gcd(f, g)·h)
        let h = rng.nonzero_zpoly(2, 5);
        assert_eq!(
            (f.clone() * h.clone()).gcd(&(g.clone() * h.clone())),
            (gg.clone() * h).normalize()
        );
        // Gauss: over Q the gcd is the monic associate of the Z[x] gcd.
        let gq = z_to_q(&f).gcd(&z_to_q(&g));
        assert_eq!(gq, z_to_q(&gg).make_monic(), "gcd over Q for {f}, {g}");
    }
}

// ---------------------------------------------------------------------------------------
// §23.5.1 GCD / LCM over Q: monic (gp: mon(p) = p/pollead(p))
// ---------------------------------------------------------------------------------------

#[test]
fn gcd_lcm_over_q_are_monic_and_match_pari() {
    // mon(gcd(2*x, x)) -> x; mon(lcm(2*x, x)) -> x (the old code returned 2x)
    assert_eq!(qi(&[0, 2]).gcd(&qi(&[0, 1])), qi(&[0, 1]));
    assert_eq!(qi(&[0, 2]).lcm(&qi(&[0, 1])), qi(&[0, 1]));
    // mon(gcd(2*x^2-2, 4*x+4)) -> x + 1; mon(lcm(...)) -> x^2 - 1
    assert_eq!(qi(&[-2, 0, 2]).gcd(&qi(&[4, 4])), qi(&[1, 1]));
    assert_eq!(qi(&[-2, 0, 2]).lcm(&qi(&[4, 4])), qi(&[-1, 0, 1]));
    // mon(gcd(x^2+1, x+1)) -> 1
    assert_eq!(qi(&[1, 0, 1]).gcd(&qi(&[1, 1])), qi(&[1]));
    // mon(gcd(0, 3*x+3)) -> x + 1
    assert_eq!(qi(&[0]).gcd(&qi(&[3, 3])), qi(&[1, 1]));
    assert_eq!(qi(&[0]).gcd(&qi(&[0])), qi(&[0]));
    // mon(gcd(x^2-1/4, x^2+x+1/4)) -> x + 1/2; mon(lcm(...)) -> x^3 + 1/2*x^2 - 1/4*x - 1/8
    let a = q(&[(-1, 4), (0, 1), (1, 1)]);
    let b = q(&[(1, 4), (1, 1), (1, 1)]);
    assert_eq!(a.gcd(&b), q(&[(1, 2), (1, 1)]));
    assert_eq!(a.lcm(&b), q(&[(-1, 8), (-1, 4), (1, 2), (1, 1)]));
    // mon(gcd(2/3*x^3 - 2/3*x, 5*x^2 + 5*x)) -> x^2 + x
    assert_eq!(
        q(&[(0, 1), (-2, 3), (0, 1), (2, 3)]).gcd(&qi(&[0, 5, 5])),
        qi(&[0, 1, 1])
    );
    // mon(gcd(Pol(1/2), Pol(3))) -> 1
    assert_eq!(q(&[(1, 2)]).gcd(&qi(&[3])), qi(&[1]));
    // mon(gcd((3*x^3-2*x+7)^2*(5*x^2+x-1)/7, (3*x^3-2*x+7)*(x^4+1)/5)) -> x^3 - 2/3*x + 7/3
    let t = qi(&[7, -2, 0, 3]);
    let a = t.clone() * t.clone() * qi(&[-1, 1, 5]) * q(&[(1, 7)]);
    let b = t * qi(&[1, 0, 0, 0, 1]) * q(&[(1, 5)]);
    assert_eq!(a.gcd(&b), q(&[(7, 3), (-2, 3), (0, 1), (1, 1)]));
}

// Content over a field is the ring-theoretic gcd: 1 for a nonzero polynomial, 0 for 0.
// (PARI's content(2/3*x+4/5) = 2/15 is the Z-module content, a different notion.)
#[test]
fn content_and_normalize_over_q() {
    assert_eq!(qi(&[6, 9, 12]).content(), rat(1, 1));
    assert_eq!(qi(&[0]).content(), rat(0, 1));
    assert_eq!(qi(&[6, 9, 12]).primitive_part(), qi(&[6, 9, 12]));
    // mon(2*x^2-2) -> x^2 - 1
    assert_eq!(qi(&[-2, 0, 2]).normalize(), qi(&[-1, 0, 1]));
    assert_eq!(q(&[(1, 3), (-2, 3)]).normalize(), q(&[(-1, 2), (1, 1)]));
    assert_eq!(qi(&[0]).normalize(), qi(&[0]));
    // Through the trait default: a unit's normalization unit is itself.
    assert_eq!(rat(2, 3).normalization_unit(), Some(rat(2, 3)));
    assert_eq!(rat(-5, 7).normalization_unit(), Some(rat(-5, 7)));
    assert_eq!(rat(0, 1).normalization_unit(), Some(rat(1, 1)));
}

// ---------------------------------------------------------------------------------------
// §23.5.1 GCD / LCM over F_p (test-local field; gp values made monic with mon)
// ---------------------------------------------------------------------------------------

#[test]
fn gcd_lcm_over_fp_match_pari() {
    // gp: gcd(f7,g7) -> Mod(1,7)*x + Mod(1,7);
    //     mon(lcm(f7,g7)) -> x^4 + 3*x^3 + 4*x^2 + 3*x + 1 (all Mod(.,7))
    let f7 = fp::<7>(&[3, 0, 5, 1]);
    let g7 = fp::<7>(&[1, 4, 3]);
    assert_eq!(f7.gcd(&g7), fp::<7>(&[1, 1]));
    assert_eq!(f7.lcm(&g7), fp::<7>(&[1, 3, 4, 3, 1]));
    // gp: gcd(x^2*Mod(1,7), 2*x*Mod(1,7)) -> Mod(2,7)*x, monic: x
    assert_eq!(fp::<7>(&[0, 0, 1]).gcd(&fp::<7>(&[0, 2])), fp::<7>(&[0, 1]));

    // gp: gcd(f2,g2) -> x + 1, lcm(f2,g2) -> x^5 + x^4 + x^2 + x, f2 = x^4+x, g2 = x^2+1 over F_2
    let f2 = fp::<2>(&[0, 1, 0, 0, 1]);
    let g2 = fp::<2>(&[1, 0, 1]);
    assert_eq!(f2.gcd(&g2), fp::<2>(&[1, 1]));
    assert_eq!(f2.lcm(&g2), fp::<2>(&[0, 1, 1, 0, 1, 1]));

    // F_5: f5 = 3x^4+2x^2+4x+1, g5 = 2x^3+x+3, h5 = 2x^2+3x+1.
    // gp: gcd(f5,g5) -> Mod(2,5), monic: 1;
    //     gcd(f5*h5, g5*h5) -> Mod(4,5)*x^2 + Mod(1,5)*x + Mod(2,5), monic: x^2 + 4*x + 3;
    //     f5*h5 -> x^6 + 4x^5 + 2x^4 + 4x^3 + x^2 + 2x + 1; g5*h5 -> 4x^5 + x^4 + 4x^3 + 4x^2 + 3
    let f5 = fp::<5>(&[1, 4, 2, 0, 3]);
    let g5 = fp::<5>(&[3, 1, 0, 2]);
    let h5 = fp::<5>(&[1, 3, 2]);
    assert_eq!(f5.gcd(&g5), fp::<5>(&[1]));
    let fh = f5 * h5.clone();
    let gh = g5 * h5;
    assert_eq!(fh, fp::<5>(&[1, 2, 1, 4, 2, 4, 1]));
    assert_eq!(gh, fp::<5>(&[3, 0, 4, 4, 1, 4]));
    assert_eq!(fh.gcd(&gh), fp::<5>(&[3, 4, 1]));
}

// ---------------------------------------------------------------------------------------
// §23.5.2 Content / PrimitivePart / ContentAndPrimitivePart and Normalize over Z
// gp: content(-6-9*x) -> 3; content(6+9*x+12*x^2) -> 3; content(-2*x-4) -> 2; content(0*x) -> 0.
// The primitive part is plain arithmetic: (-6-9x)/3 = -2-3x (MAGMA "p divided by its content").
// ---------------------------------------------------------------------------------------

#[test]
fn content_primitive_part_normalize_over_z() {
    assert_eq!(z(&[-6, -9]).content(), Integer::from(3));
    assert_eq!(z(&[6, 9, 12]).content(), Integer::from(3));
    assert_eq!(z(&[-4, -2]).content(), Integer::from(2));
    assert_eq!(z(&[0]).content(), Integer::from(0));
    assert_eq!(z(&[-6, -9]).primitive_part(), z(&[-2, -3]));
    assert_eq!(z(&[0]).primitive_part(), z(&[0]));
    assert_eq!(
        z(&[6, 4]).content_and_primitive_part(),
        (Integer::from(2), z(&[3, 2]))
    );
    assert_eq!(
        z(&[0]).content_and_primitive_part(),
        (Integer::from(0), z(&[0]))
    );
    assert_eq!(z(&[-6, -9]).normalize(), z(&[6, 9]));
    assert_eq!(z(&[-6, 9]).normalize(), z(&[-6, 9]));
    assert_eq!(z(&[0]).normalize(), z(&[0]));
}

// gp: issquarefree(x^2+x) -> 1 (formerly hung); issquarefree(x^3-x^2-x+1) -> 0
#[test]
fn is_square_free_over_z() {
    assert!(within(5, || z(&[0, 1, 1]).is_square_free()));
    assert!(!within(5, || z(&[1, -1, -1, 1]).is_square_free()));
}

// gp: factor(-x^3+4*x^2-5*x+2) -> [x - 2, 1; x - 1, 2] (up to the unit -1)
#[test]
fn squarefree_decomposition_over_q_multiplicities() {
    let p = qi(&[2, -5, 4, -1]);
    let mut d: Vec<(QPoly, usize)> = p
        .squarefree_decomposition()
        .into_iter()
        .map(|(f, m)| (f.make_monic(), m))
        .collect();
    d.sort_by_key(|(_, m)| *m);
    assert_eq!(d, vec![(qi(&[-2, 1]), 1), (qi(&[-1, 1]), 2)]);
}
