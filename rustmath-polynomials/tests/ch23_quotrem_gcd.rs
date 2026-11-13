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
//! Two conventions are RustMath's own and **not verified against MAGMA** (the current handbook
//! and the online calculator need a login, and no MAGMA was available): which minimal-degree
//! remainder `Quotrem` returns over Z when the leading coefficient of the divisor is not a
//! unit, and `Content` over a field (1 for a nonzero polynomial). The tests below pin the
//! RustMath values; they are not MAGMA reference values. Evidence gathered so far: the V2.19
//! handbook (§23.4.10, §23.5.2) has the V2.3 text word for word and gives no example. Sage
//! over ZZ calls FLINT's `fmpz_poly_divrem` (`polynomial_integer_dense_flint.pyx`, `quo_rem`
//! doctests in the develop source on GitHub: `(5*x+7).quo_rem(3) == (x + 2, 2*x + 1)`,
//! `(x^2).quo_rem(2*x - 1) == (0, x^2)`), another remainder of the same minimal degree. Sage's
//! `content_ideal` over QQ is the unit ideal (`QQ.ideal(2)` is "Principal ideal (1) of
//! Rational Field"), which agrees with `Content = 1`.
//!
//! Inputs marked "formerly hung" looped forever before the fix; they run inside `within`, so
//! a regression fails fast instead of stalling the test run. The same holds for the gcds over
//! Q of degree 40 to 50, which plain Euclid over Q (the earlier algorithm) did not finish in
//! minutes; they go through Z[x] by Gauss' lemma now.

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

/// Run `f` on a thread and fail if it does not finish within `secs` seconds. A panic inside
/// `f` (a failed assertion) is re-raised here with its own message, not reported as a timeout.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(secs)) {
        Ok(v) => v,
        // The sender was dropped without sending: `f` panicked.
        Err(mpsc::RecvTimeoutError::Disconnected) => match worker.join() {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(()) => unreachable!("the worker returned without sending its result"),
        },
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("timed out after {secs} s: a non-terminating or blown-up computation is back")
        }
    }
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
// Division in F_p is multiplication by the inverse, which this lint cannot see.
#[allow(clippy::suspicious_arithmetic_impl)]
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
// coefficient that lc(g) does not divide (RustMath convention, MAGMA unverified). Where every
// step is exact the result is the Z-image of PARI's divrem; otherwise the identity
// f = q·g + r is checked with gp.
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

// Over Z the minimal-degree remainder is not unique. RustMath (convention, MAGMA unverified):
// div_rem(3x^2+x, 2x) = (0, 3x^2+x). FLINT fmpz_poly_divrem (python-flint 0.9.0,
// divmod(3*x^2+x, 2*x)): (x, x^2+x). Both satisfy MAGMA's contract "deg r minimal": the x^2
// coefficient of f - q'·g is 3 - 2t, odd for every integer t, so every remainder has degree 2.
// Likewise -3x^2 by 2x: RustMath (0, -3x^2), FLINT (-2x, x^2).
#[test]
fn quotrem_over_z_minimal_remainder_is_not_unique() {
    let (f, g) = (z(&[0, 1, 3]), z(&[0, 2]));
    assert_eq!(f.div_rem(&g).unwrap(), (z(&[0]), z(&[0, 1, 3])));
    let (qf, rf) = (z(&[0, 1]), z(&[0, 1, 1])); // FLINT
    assert_eq!(qf * g.clone() + rf.clone(), f);
    assert_eq!(rf.degree(), f.div_rem(&g).unwrap().1.degree()); // both degree 2 = minimal
    assert_eq!(
        z(&[0, 0, -3]).div_rem(&g).unwrap(),
        (z(&[0]), z(&[0, 0, -3]))
    );
    assert_eq!(z(&[0, -2]) * g.clone() + z(&[0, 0, 1]), z(&[0, 0, -3])); // FLINT
    // Sage/FLINT doctests (polynomial_integer_dense_flint.pyx, quo_rem):
    // (5*x+7).quo_rem(3) == (x + 2, 2*x + 1) and (x^2).quo_rem(2*x - 1) == (0, x^2).
    let (f5, three) = (z(&[7, 5]), z(&[3]));
    assert_eq!(f5.div_rem(&three).unwrap(), (z(&[0]), z(&[7, 5])));
    let (qs, rs) = (z(&[2, 1]), z(&[1, 2])); // FLINT
    assert_eq!(qs * three.clone() + rs.clone(), f5);
    assert_eq!(rs.degree(), Some(1)); // the minimal degree, as ours
    assert_eq!(
        z(&[0, 0, 1]).div_rem(&z(&[-1, 2])).unwrap(),
        (z(&[0]), z(&[0, 0, 1]))
    );
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

/// Invariants of MAGMA `Quotrem` on 300 pseudo-random pairs over Z. Most pairs have a
/// non-unit lc(g), the inexact-step case that formerly hung, so each pair runs inside `within`.
#[test]
fn quotrem_over_z_invariants() {
    let mut rng = Lcg(0x5eed_0023);
    for _ in 0..300 {
        let f = rng.zpoly(6, 9);
        let g = rng.nonzero_zpoly(6, 9);
        within(10, move || {
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
        });
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
/// pseudo-random pairs of nonzero polynomials with a common factor. The gcd over Z hung
/// before the fix, so each pair runs inside `within`.
#[test]
fn gcd_over_z_agrees_with_zx_and_satisfies_identities() {
    let mut rng = Lcg(0x0023_0005_0001);
    for _ in 0..300 {
        let common = rng.nonzero_zpoly(3, 6);
        let f = rng.nonzero_zpoly(5, 20) * common.clone();
        let g = rng.nonzero_zpoly(5, 20) * common.clone();
        // Drawn here, in the same order as before, so the 300 cases are unchanged.
        let h = rng.nonzero_zpoly(2, 5);
        within(10, move || {
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
            assert_eq!(
                (f.clone() * h.clone()).gcd(&(g.clone() * h.clone())),
                (gg.clone() * h).normalize()
            );
            // Gauss: over Q the gcd is the monic associate of the Z[x] gcd.
            let gq = z_to_q(&f).gcd(&z_to_q(&g));
            assert_eq!(gq, z_to_q(&gg).make_monic(), "gcd over Q for {f}, {g}");
        });
    }
}

/// `gcd(f, 0)` and `gcd(0, f)` keep the content and agree with `zx::subresultant_gcd` on 50
/// pseudo-random f, zero included (gp: gcd(4*x+4, 0) -> 4*x + 4; the MAGMA-normalized value
/// has a positive leading coefficient). `UnivariatePolynomial` stores 0 as `[0]` and `zx` as
/// `[]`, so both sides go through `UnivariatePolynomial::new`.
#[test]
fn gcd_with_zero_agrees_with_zx_and_keeps_the_content() {
    let zero = z(&[0]);
    let empty: Vec<Integer> = Vec::new();
    let mut rng = Lcg(0x0023_0005_0002);
    let mut inputs = vec![z(&[0]), z(&[4, 4]), z(&[-4, -4]), z(&[-6]), z(&[9, 0, -6])];
    for _ in 0..45 {
        inputs.push(rng.zpoly(6, 30));
    }
    for f in inputs {
        let a = f.gcd(&zero);
        let b = zero.gcd(&f);
        let r1 = ZPoly::new(zx::subresultant_gcd(f.coefficients(), &empty));
        let r2 = ZPoly::new(zx::subresultant_gcd(&empty, f.coefficients()));
        assert_eq!(a, r1, "gcd({f}, 0) vs zx");
        assert_eq!(b, r2, "gcd(0, {f}) vs zx");
        assert_eq!(a, f.normalize(), "gcd({f}, 0) = Normalize({f})");
        assert_eq!(a.content(), f.content(), "gcd({f}, 0) keeps the content");
    }
    assert_eq!(z(&[4, 4]).gcd(&zero), z(&[4, 4]));
    assert_eq!(z(&[9, 0, -6]).gcd(&zero), z(&[-9, 0, 6]));
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

impl Lcg {
    /// `n/d` with `n` in [-nb, nb], then `d` in [1, db] (gp `rat` below).
    fn rational(&mut self, nb: i64, db: i64) -> Rational {
        let n = self.range(-nb, nb);
        let d = self.range(1, db);
        rat(n, d)
    }
    /// A polynomial over Q of exact degree `deg` (gp `qpol` below): `deg` coefficients from
    /// [`Lcg::rational`], low degree first, then a leading coefficient `n/d` with `n` in
    /// [1, nb] and `d` in [1, db].
    fn qpoly_exact(&mut self, deg: usize, nb: i64, db: i64) -> QPoly {
        let mut c: Vec<Rational> = (0..deg).map(|_| self.rational(nb, db)).collect();
        let n = self.range(1, nb);
        let d = self.range(1, db);
        c.push(rat(n, d));
        UnivariatePolynomial::new(c)
    }
    /// A polynomial over Q of degree at most `max_deg` (possibly 0 or a constant).
    fn qpoly(&mut self, max_deg: i64, nb: i64, db: i64) -> QPoly {
        let d = self.range(0, max_deg) as usize;
        UnivariatePolynomial::new((0..=d).map(|_| self.rational(nb, db)).collect())
    }
}

/// Seconds allowed for one degree >= 40 pair over Q with all its checks (four gcds, an lcm and
/// two exact quotients). In a debug build the Gauss-lemma path through Z[x] takes about 30 ms
/// per gcd at degree 42 and about 0.6 s for all five pairs; plain Euclid over Q, the earlier
/// algorithm, took 327 s for the first pair's gcd (about 6.5 s at degree 25, and more than
/// 4 minutes at degree 50).
const Q_GCD_SECS: u64 = 30;

/// MAGMA `GCD` over Q[x] is monic. Pairs `f = a·c^rep`, `g = b·c` of degree 41 to 50 with
/// random rational coefficients (numerators in [-9, 9], denominators in [1, 9]), generated by
/// the `Lcg` in Rust and by the same LCG in gp:
///
/// ```text
/// s = 0;
/// {nxt() = s = (s * 6364136223846793005 + 1442695040888963407) % 2^64; s >> 33;}
/// {rng(lo, hi) = lo + nxt() % (hi - lo + 1);}
/// {rat(nb, db) = my(n, d); n = rng(-nb, nb); d = rng(1, db); n / d;}
/// {qpol(deg, nb, db) = my(c, n, d); c = vector(deg + 1);
///   for (i = 1, deg, c[i] = rat(nb, db));
///   n = rng(1, nb); d = rng(1, db); c[deg + 1] = n / d; Pol(Vecrev(c));}
/// mon(p) = p / pollead(p);
/// \\ for each case: s = seed; c = qpol(dc,9,9); a = qpol(da,9,9); b = qpol(db,9,9);
/// \\ f = a*c^rep; g = b*c; G = mon(gcd(f, g)); Vecrev(G)
/// ```
///
/// gp prints `subst(f, x, 1)`, `subst(f, x, -1)` and `pollead(f)` (and the same for g), which
/// pin the generated inputs, and `Vecrev(mon(gcd(f, g)))`. Cross-checks in gp: in every case
/// `mon(gcd(f, g)) == mon(c)` (and `gcd(a·c^(rep-1), b) = 1`), and
/// `mon(lcm(f, g)) == mon(f*g/G)`.
#[test]
fn gcd_over_q_degree_40_and_up_matches_pari_within_time_bound() {
    type Pin = (i64, i64);
    struct Case {
        seed: u64,
        dc: usize,
        da: usize,
        db: usize,
        rep: u32,
        // (f(1), f(-1), lc f, g(1), g(-1), lc g)
        pins: [Pin; 6],
        gcd: Vec<Pin>,
    }
    let cases = vec![
        // deg f = 42, deg g = 41, deg gcd = 6
        Case {
            seed: 0x0023_0005_0051_0001,
            dc: 6,
            da: 36,
            db: 35,
            rep: 1,
            pins: [
                (-1143378713, 6350400),
                (253266893, 1270080),
                (20, 9),
                (-175724501, 3175200),
                (335815001, 3175200),
                (1, 2),
            ],
            gcd: vec![
                (18, 1),
                (-27, 28),
                (-63, 32),
                (-3, 2),
                (27, 4),
                (-9, 5),
                (1, 1),
            ],
        },
        // deg f = 48, deg g = 45, deg gcd = 8
        Case {
            seed: 0x0023_0005_0051_0002,
            dc: 8,
            da: 40,
            db: 37,
            rep: 1,
            pins: [
                (212629, 12960),
                (140287, 6480),
                (4, 1),
                (610111, 6480),
                (-51947, 1080),
                (2, 1),
            ],
            gcd: vec![
                (-5, 32),
                (7, 32),
                (1, 4),
                (-5, 36),
                (-2, 5),
                (1, 4),
                (2, 5),
                (1, 12),
                (1, 1),
            ],
        },
        // deg f = 50, deg g = 49, deg gcd = 5
        Case {
            seed: 0x0023_0005_0051_0003,
            dc: 5,
            da: 45,
            db: 44,
            rep: 1,
            pins: [
                (-41912981, 453600),
                (-2832581, 453600),
                (5, 18),
                (12990823, 75600),
                (345539, 113400),
                (40, 81),
            ],
            gcd: vec![(-9, 40), (-18, 5), (-9, 2), (-54, 25), (-6, 5), (1, 1)],
        },
        // deg f = 45, deg g = 44, deg gcd = 20
        Case {
            seed: 0x0023_0005_0051_0004,
            dc: 20,
            da: 25,
            db: 24,
            rep: 1,
            pins: [
                (506411, 7840),
                (224807, 3528),
                (7, 10),
                (-70993, 1960),
                (-144364, 441),
                (14, 15),
            ],
            gcd: vec![
                (9, 14),
                (18, 7),
                (3, 4),
                (27, 14),
                (36, 49),
                (24, 49),
                (-3, 2),
                (6, 49),
                (-24, 7),
                (3, 7),
                (6, 7),
                (27, 28),
                (8, 7),
                (54, 7),
                (3, 7),
                (-15, 14),
                (-9, 14),
                (0, 1),
                (18, 7),
                (-27, 7),
                (1, 1),
            ],
        },
        // f = a·c^2: deg f = 44, deg g = 47, deg gcd = 7
        Case {
            seed: 0x0023_0005_0051_0005,
            dc: 7,
            da: 30,
            db: 40,
            rep: 2,
            pins: [
                (502897, 90720),
                (-845543, 4480),
                (25, 36),
                (-542903, 10080),
                (-63263, 1120),
                (10, 21),
            ],
            gcd: vec![
                (-7, 5),
                (0, 1),
                (0, 1),
                (6, 5),
                (-3, 1),
                (-6, 5),
                (-3, 4),
                (1, 1),
            ],
        },
    ];
    for case in cases {
        let mut rng = Lcg(case.seed);
        let c = rng.qpoly_exact(case.dc, 9, 9);
        let a = rng.qpoly_exact(case.da, 9, 9);
        let b = rng.qpoly_exact(case.db, 9, 9);
        let f = a * c.pow(case.rep);
        let g = b * c;
        assert!(f.degree() >= Some(40) || g.degree() >= Some(40));
        let one = rat(1, 1);
        let pins = [
            f.evaluate(&one),
            f.evaluate(&-one.clone()),
            f.leading_coefficient().unwrap().clone(),
            g.evaluate(&one),
            g.evaluate(&-one.clone()),
            g.leading_coefficient().unwrap().clone(),
        ];
        let expected_pins: Vec<Rational> = case.pins.iter().map(|&(n, d)| rat(n, d)).collect();
        assert_eq!(
            pins.to_vec(),
            expected_pins,
            "seed {:#x}: generated inputs",
            case.seed
        );
        let expected = q(&case.gcd);
        let seed = case.seed;
        within(Q_GCD_SECS, move || {
            let gg = f.gcd(&g);
            assert_eq!(gg, expected, "seed {seed:#x}: gcd vs PARI mon(gcd(f, g))");
            assert_eq!(g.gcd(&f), expected, "seed {seed:#x}: gcd(g, f)");
            assert!(gg.is_monic());
            // Cross-checks: gcd | f, gcd | g, and the cofactors are coprime.
            let cf = f.exact_quotient(&gg).expect("gcd | f");
            let cg = g.exact_quotient(&gg).expect("gcd | g");
            assert_eq!(cf.gcd(&cg), qi(&[1]), "seed {seed:#x}: cofactors coprime");
            // gp: mon(lcm(f, g)) == mon(f*g/G)
            assert_eq!(
                f.lcm(&g),
                (cf * g.clone()).make_monic(),
                "seed {seed:#x}: lcm"
            );
        });
    }
}

/// Q through `Rational` arithmetic, but without the [`EuclideanDomain::as_any`] hook, so its
/// polynomial gcd takes the generic field path (Euclid over Q, then `normalize`), which the
/// Gauss-lemma path for `Rational` must agree with.
#[derive(Debug, Clone, PartialEq)]
struct QGen(Rational);
impl fmt::Display for QGen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Add for QGen {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        QGen(self.0 + o.0)
    }
}
impl Sub for QGen {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        QGen(self.0 - o.0)
    }
}
impl Mul for QGen {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        QGen(self.0 * o.0)
    }
}
impl Neg for QGen {
    type Output = Self;
    fn neg(self) -> Self {
        QGen(-self.0)
    }
}
impl Ring for QGen {
    fn zero() -> Self {
        QGen(Rational::zero())
    }
    fn one() -> Self {
        QGen(Rational::one())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
}
impl CommutativeRing for QGen {}
impl IntegralDomain for QGen {}
impl EuclideanDomain for QGen {
    fn norm(&self) -> u64 {
        u64::from(!self.0.is_zero())
    }
    fn div_rem(&self, other: &Self) -> Result<(Self, Self)> {
        if other.0.is_zero() {
            return Err(MathError::DivisionByZero);
        }
        Ok((
            QGen(self.0.clone() / other.0.clone()),
            QGen(Rational::zero()),
        ))
    }
}

fn to_qgen(p: &QPoly) -> UnivariatePolynomial<QGen> {
    UnivariatePolynomial::new(p.coefficients().iter().cloned().map(QGen).collect())
}

/// Differential test: on 300 small pseudo-random pairs over Q (zero, constants, common factors,
/// negative leading coefficients), the Gauss-lemma gcd and lcm for `Rational` coefficients equal
/// those of the generic field path (`QGen`, the same field without the hook).
#[test]
fn gcd_over_q_agrees_with_the_generic_field_path() {
    let mut rng = Lcg(0x0023_0005_0052_0001);
    for i in 0..300 {
        let common = rng.qpoly(3, 6, 6);
        let mut f = rng.qpoly(5, 9, 9);
        let mut g = rng.qpoly(5, 9, 9);
        if i % 3 != 0 {
            f = f * common.clone();
            g = g * common.clone();
        }
        let (f2, g2) = (f.clone(), g.clone());
        let (gg, ll) = within(10, move || (f2.gcd(&g2), f2.lcm(&g2)));
        let (fq, gq) = (to_qgen(&f), to_qgen(&g));
        assert_eq!(to_qgen(&gg), fq.gcd(&gq), "gcd({f}, {g})");
        assert_eq!(to_qgen(&ll), fq.lcm(&gq), "lcm({f}, {g})");
        assert_eq!(gg, g.gcd(&f), "gcd({g}, {f})");
        if !gg.is_zero() {
            assert!(gg.is_monic(), "gcd({f}, {g}) is monic");
        }
    }
}

/// Edge cases of the Gauss-lemma gcd over Q. gp: mon(gcd(-3*x^2+3, 6*x-6)) -> x - 1;
/// mon(gcd(Pol(-2/3), x^2+1)) -> 1; mon(gcd(-1/2*x^3+1/2*x, x^2-x)) -> x^2 - x;
/// mon(gcd(f, -f)) for f = -2/3*x^2+4/5 -> x^2 - 6/5; mon(gcd(x^4-1/16, 2*x^2-1/2)) -> x^2 - 1/4.
#[test]
fn gcd_over_q_edge_cases_match_pari() {
    assert_eq!(qi(&[3, 0, -3]).gcd(&qi(&[-6, 6])), qi(&[-1, 1]));
    assert_eq!(q(&[(-2, 3)]).gcd(&qi(&[1, 0, 1])), qi(&[1]));
    assert_eq!(
        q(&[(0, 1), (1, 2), (0, 1), (-1, 2)]).gcd(&qi(&[0, -1, 1])),
        qi(&[0, -1, 1])
    );
    let f = q(&[(4, 5), (0, 1), (-2, 3)]);
    assert_eq!(f.gcd(&-f.clone()), q(&[(-6, 5), (0, 1), (1, 1)]));
    assert_eq!(f.gcd(&f), q(&[(-6, 5), (0, 1), (1, 1)]));
    assert_eq!(
        q(&[(-1, 16), (0, 1), (0, 1), (0, 1), (1, 1)]).gcd(&q(&[(-1, 2), (0, 1), (2, 1)])),
        q(&[(-1, 4), (0, 1), (1, 1)])
    );
}

// Content over a field is the ring-theoretic gcd: 1 for a nonzero polynomial, 0 for 0
// (convention, MAGMA unverified; see UnivariatePolynomial::content).
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

// ---------------------------------------------------------------------------------------
// Free functions in `factorization` (content, primitive_part, square_free_factorization,
// is_square_free) and the method `is_square_free`, against PARI. gp -q:
//   content(-6-9*x) -> 3; content(6-9*x) -> 3; content(0*x) -> 0;
//   (6-9*x)/content(6-9*x) -> -3*x + 2; content((2*x^2+3)*Mod(1,5)) -> Mod(1, 5)
// ---------------------------------------------------------------------------------------

#[test]
fn free_content_and_primitive_part_agree_with_the_methods() {
    use rustmath_polynomials::factorization;
    for f in [
        z(&[-6, -9]),
        z(&[6, -9]),
        z(&[0]),
        z(&[6, 9, 12]),
        z(&[-4, -2]),
        z(&[-5]),
    ] {
        assert_eq!(factorization::content(&f), f.content(), "content({f})");
        assert_eq!(
            factorization::primitive_part(&f),
            f.primitive_part(),
            "pp({f})"
        );
    }
    assert_eq!(factorization::content(&z(&[-6, -9])), Integer::from(3));
    assert_eq!(factorization::content(&z(&[6, -9])), Integer::from(3));
    assert_eq!(factorization::content(&z(&[0])), Integer::from(0));
    assert_eq!(factorization::primitive_part(&z(&[6, -9])), z(&[2, -3]));
    assert_eq!(factorization::primitive_part(&z(&[0])), z(&[0]));
    // Over a field: content 1, primitive part the polynomial itself (was 2 and x^2 + 4).
    assert_eq!(
        factorization::content(&fp::<5>(&[3, 0, 2])),
        Fp::<5>::new(1)
    );
    assert_eq!(
        factorization::primitive_part(&fp::<5>(&[3, 0, 2])),
        fp::<5>(&[3, 0, 2])
    );
    // Over Q: 1 (convention, MAGMA unverified), was the last coefficient 12.
    assert_eq!(factorization::content(&qi(&[6, 9, 12])), rat(1, 1));
}

// gp: issquarefree(4*x+4) -> 1; issquarefree(2*x+2) -> 1; issquarefree(Pol(4)) -> 1;
//     issquarefree(Pol(5)) -> 1; issquarefree(Pol(0)) -> 0; issquarefree(x^2-2*x+1) -> 0;
//     issquarefree(x^2+x) -> 1; issquarefree(6*x^2+12*x+6) -> 0;
//     issquarefree(12*(x^2+1)^3*(x-3)^2*(2*x+1)) -> 0
//     (Vecrev: [108, 144, 192, 456, -72, 504, -288, 216, -132, 24]);
//     issquarefree(x^3*Mod(1,3)) -> 0; issquarefree((x^3+2)*Mod(1,3)) -> 0;
//     issquarefree((x^2+2)*Mod(1,3)) -> 1;
//     issquarefree((x+2)^2*(x^2+1)*Mod(1,5)) -> 0 (Vecrev lift: [4, 4, 0, 4, 1]).
// The content is not examined (4x + 4 = 2^2 (x + 1) is square-free in this sense).
#[test]
fn is_square_free_matches_pari_method_and_free_function() {
    use rustmath_polynomials::factorization;
    let zcases: Vec<(ZPoly, bool)> = vec![
        (z(&[4, 4]), true),
        (z(&[2, 2]), true),
        (z(&[4]), true),
        (z(&[5]), true),
        (z(&[0]), false),
        (z(&[1, -2, 1]), false),
        (z(&[0, 1, 1]), true),
        (z(&[6, 12, 6]), false),
        (
            z(&[108, 144, 192, 456, -72, 504, -288, 216, -132, 24]),
            false,
        ),
    ];
    for (f, want) in zcases {
        assert_eq!(f.is_square_free(), want, "is_square_free({f})");
        assert_eq!(
            factorization::is_square_free(&f),
            Ok(want),
            "free is_square_free({f})"
        );
    }
    let f3: Vec<(UnivariatePolynomial<Fp<3>>, bool)> = vec![
        (fp::<3>(&[0, 0, 0, 1]), false),
        (fp::<3>(&[2, 0, 0, 1]), false),
        (fp::<3>(&[2, 0, 1]), true),
    ];
    for (f, want) in f3 {
        assert_eq!(f.is_square_free(), want, "is_square_free({f}) over GF(3)");
        assert_eq!(factorization::is_square_free(&f), Ok(want));
    }
    let f5 = fp::<5>(&[4, 4, 0, 4, 1]);
    assert!(!f5.is_square_free());
    assert_eq!(factorization::is_square_free(&f5), Ok(false));
}

// gp: factor(-2*(x-1)^2*(x+2)) -> [x - 1, 2; x + 2, 1]   (Vecrev: [-4, 6, 0, -2])
//     factor(-x^3+4*x^2-5*x+2) -> [x - 2, 1; x - 1, 2]
//     factor(12*(x^2+1)^3*(x-3)^2*(2*x+1)) -> [x - 3, 2; 2*x + 1, 1; x^2 + 1, 3]
//     factormod((x+1)^2*(x+2),3) -> [x + 1, 2; x + 2, 1]          (Vecrev lift: [2, 2, 1, 1])
//     factormod((x+1)^3*(x+2),3) -> [x + 1, 3; x + 2, 1]          (Vecrev lift: [2, 1, 0, 2, 1])
//     factormod((x+1)^2*(x+3)^3*(x^2+2),5) -> [x + 1, 2; x + 3, 3; x^2 + 2, 1]
//                                                   (Vecrev lift: [4, 2, 2, 3, 2, 3, 1, 1])
//     factormod(x^3,3) -> [x, 3]
// Factors are normalized (positive leading coefficient over Z, monic over a field), sorted
// by multiplicity; content and unit are dropped.
#[test]
fn free_square_free_factorization_matches_pari() {
    use rustmath_polynomials::factorization::square_free_factorization;
    assert_eq!(
        square_free_factorization(&z(&[-4, 6, 0, -2])).unwrap(),
        vec![(z(&[2, 1]), 1), (z(&[-1, 1]), 2)]
    );
    assert_eq!(
        square_free_factorization(&qi(&[2, -5, 4, -1])).unwrap(),
        vec![(qi(&[-2, 1]), 1), (qi(&[-1, 1]), 2)]
    );
    let big = z(&[108, 144, 192, 456, -72, 504, -288, 216, -132, 24]);
    let d = square_free_factorization(&big).unwrap();
    assert_eq!(
        d,
        vec![(z(&[1, 2]), 1), (z(&[-3, 1]), 2), (z(&[1, 0, 1]), 3)]
    );
    let prod = d.iter().fold(z(&[1]), |acc, (f, m)| {
        (0..*m).fold(acc, |a, _| a * f.clone())
    });
    assert_eq!(prod, big.primitive_part().normalize());
    // Characteristic p, multiplicities below p: correct.
    assert_eq!(
        square_free_factorization(&fp::<3>(&[2, 2, 1, 1])).unwrap(),
        vec![(fp::<3>(&[2, 1]), 1), (fp::<3>(&[1, 1]), 2)]
    );
    assert_eq!(
        square_free_factorization(&fp::<5>(&[4, 2, 2, 3, 2, 3, 1, 1])).unwrap(),
        vec![
            (fp::<5>(&[2, 0, 1]), 1),
            (fp::<5>(&[1, 1]), 2),
            (fp::<5>(&[3, 1]), 3)
        ]
    );
    // A multiplicity divisible by p: an honest Err (was Ok([(x+2, 1), (x+2, 1)])).
    assert!(matches!(
        square_free_factorization(&fp::<3>(&[2, 1, 0, 2, 1])),
        Err(MathError::NotSupported(_))
    ));
    assert!(matches!(
        square_free_factorization(&fp::<3>(&[0, 0, 0, 1])),
        Err(MathError::NotSupported(_))
    ));
    // Constants and 0: no factor of positive degree.
    assert_eq!(square_free_factorization(&z(&[5])).unwrap(), vec![]);
    assert_eq!(square_free_factorization(&z(&[0])).unwrap(), vec![]);
}

// The method squarefree_decomposition (over a field) in characteristic p: correct while no
// multiplicity is divisible by p, a panic otherwise (formerly wrong values: (x+1)^3 (x+2) over
// GF(3) gave [(x+2, 1), (x^3+1, 2)], degree 7 for a degree-4 input; x^3 gave [(x^3, 1)]).
// gp: factormod((x+1)^2*(x+3)^3*(x^2+2),5) -> [x + 1, 2; x + 3, 3; x^2 + 2, 1]
#[test]
fn squarefree_decomposition_method_in_characteristic_p() {
    assert_eq!(
        fp::<3>(&[2, 2, 1, 1]).squarefree_decomposition(),
        vec![(fp::<3>(&[2, 1]), 1), (fp::<3>(&[1, 1]), 2)]
    );
    assert_eq!(
        fp::<5>(&[4, 2, 2, 3, 2, 3, 1, 1]).squarefree_decomposition(),
        vec![
            (fp::<5>(&[2, 0, 1]), 1),
            (fp::<5>(&[1, 1]), 2),
            (fp::<5>(&[3, 1]), 3)
        ]
    );
}

#[test]
#[should_panic(expected = "a multiplicity is divisible by the characteristic")]
fn squarefree_decomposition_method_panics_on_multiplicity_p() {
    // gp: factormod((x+1)^3*(x+2),3) -> [x + 1, 3; x + 2, 1]
    let _ = fp::<3>(&[2, 1, 0, 2, 1]).squarefree_decomposition();
}

#[test]
#[should_panic(expected = "f' = 0 in positive characteristic")]
fn squarefree_decomposition_method_panics_on_zero_derivative() {
    // gp: factormod(x^3,3) -> [x, 3]
    let _ = fp::<3>(&[0, 0, 0, 1]).squarefree_decomposition();
}
