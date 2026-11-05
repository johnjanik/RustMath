//! Tate's algorithm for elliptic curves over Q
//!
//! Given an integral Weierstrass model of an elliptic curve E/Q and a prime
//! p, Tate's algorithm computes the local reduction data of E at p:
//!
//! * the Kodaira type of the special fibre of the Néron minimal model,
//! * the exponent f_p of p in the conductor of E,
//! * the local Tamagawa number c_p = [E(Q_p) : E^0(Q_p)],
//! * a p-minimal integral Weierstrass model, and
//! * the valuation of the minimal discriminant at p.
//!
//! The implementation follows the standard step 1-10 formulation of the
//! algorithm (Silverman, *Advanced Topics in the Arithmetic of Elliptic
//! Curves*, IV.9; Cremona, *Algorithms for Modular Elliptic Curves*, §3.2;
//! the structure of the case analysis matches Sage's
//! `ell_local_data._tate` and PARI's `elllocalred`). All arithmetic is
//! exact over Z; the residue-field root findings at p = 2 and p = 3 use the
//! characteristic-specific formulas (sqrt = identity on F_2, cube root =
//! identity on F_3), so the algorithm is correct at the wild primes 2 and 3
//! as well as at tame primes.
//!
//! Every branch of this implementation was validated against PARI/GP's
//! `elllocalred` / `ellglobalred` on curves covering all Kodaira types
//! (see the tests at the bottom of this file, each of which cites its
//! ground truth).

use crate::curve::EllipticCurve;
use rustmath_integers::prime::{factor, is_prime};
use rustmath_integers::Integer;
use std::fmt;

/// Kodaira type of the special fibre of the Néron minimal model at a prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KodairaSymbol {
    /// I_n for n >= 0. `In(0)` is good reduction ("I0"); `In(n)` with n >= 1
    /// is multiplicative reduction.
    In(u32),
    /// Type II (additive).
    II,
    /// Type III (additive).
    III,
    /// Type IV (additive).
    IV,
    /// I_n^* for n >= 0 (additive). `InStar(0)` is "I0*".
    InStar(u32),
    /// Type IV* (additive).
    IVStar,
    /// Type III* (additive).
    IIIStar,
    /// Type II* (additive).
    IIStar,
}

impl fmt::Display for KodairaSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KodairaSymbol::In(n) => write!(f, "I{}", n),
            KodairaSymbol::II => write!(f, "II"),
            KodairaSymbol::III => write!(f, "III"),
            KodairaSymbol::IV => write!(f, "IV"),
            KodairaSymbol::InStar(n) => write!(f, "I{}*", n),
            KodairaSymbol::IVStar => write!(f, "IV*"),
            KodairaSymbol::IIIStar => write!(f, "III*"),
            KodairaSymbol::IIStar => write!(f, "II*"),
        }
    }
}

/// The reduction type of an elliptic curve at a prime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionType {
    /// Good reduction (Kodaira I0, f_p = 0).
    Good,
    /// Split multiplicative reduction (Kodaira I_n, n >= 1, f_p = 1; the
    /// tangent directions at the node are rational over F_p).
    SplitMultiplicative,
    /// Non-split multiplicative reduction (Kodaira I_n, n >= 1, f_p = 1).
    NonsplitMultiplicative,
    /// Additive reduction (all other Kodaira types, f_p >= 2).
    Additive,
}

/// Local reduction data of an elliptic curve over Q at a prime p, as
/// computed by Tate's algorithm.
#[derive(Debug, Clone)]
pub struct LocalData {
    /// The prime p.
    pub prime: Integer,
    /// Kodaira type of the special fibre at p.
    pub kodaira: KodairaSymbol,
    /// The exponent f_p of p in the conductor N of E.
    pub conductor_exponent: u32,
    /// The local Tamagawa number c_p = [E(Q_p) : E^0(Q_p)].
    pub tamagawa_number: u32,
    /// An integral model of E that is minimal at p. NOTE: this model is
    /// only guaranteed minimal at p (it is obtained from the input model by
    /// the coordinate changes the algorithm performs); it is not the
    /// globally reduced minimal model, and its coefficients are not
    /// normalized.
    pub minimal_model: EllipticCurve,
    /// v_p of the discriminant of the p-minimal model.
    pub minimal_disc_valuation: u32,
    /// Reduction type at p (good / split or non-split multiplicative /
    /// additive).
    pub reduction: ReductionType,
}

/// Working Weierstrass equation with integer coefficients.
#[derive(Clone)]
struct WEq {
    a1: Integer,
    a2: Integer,
    a3: Integer,
    a4: Integer,
    a6: Integer,
}

impl WEq {
    fn b2(&self) -> Integer {
        &self.a1 * &self.a1 + Integer::from(4) * self.a2.clone()
    }
    fn b4(&self) -> Integer {
        Integer::from(2) * self.a4.clone() + &self.a1 * &self.a3
    }
    fn b6(&self) -> Integer {
        &self.a3 * &self.a3 + Integer::from(4) * self.a6.clone()
    }
    fn b8(&self) -> Integer {
        self.a1.clone() * self.a1.clone() * self.a6.clone()
            + Integer::from(4) * self.a2.clone() * self.a6.clone()
            - self.a1.clone() * self.a3.clone() * self.a4.clone()
            + self.a2.clone() * self.a3.clone() * self.a3.clone()
            - self.a4.clone() * self.a4.clone()
    }
    fn c4(&self) -> Integer {
        let b2 = self.b2();
        &b2 * &b2 - Integer::from(24) * self.b4()
    }
    fn c6(&self) -> Integer {
        let b2 = self.b2();
        -(b2.clone() * b2.clone() * b2.clone()) + Integer::from(36) * b2 * self.b4()
            - Integer::from(216) * self.b6()
    }
    fn disc(&self) -> Integer {
        let b2 = self.b2();
        let b4 = self.b4();
        let b6 = self.b6();
        let b8 = self.b8();
        -(b2.clone() * b2.clone() * b8)
            - Integer::from(8) * b4.clone() * b4.clone() * b4.clone()
            - Integer::from(27) * b6.clone() * b6.clone()
            + Integer::from(9) * b2 * b4 * b6
    }

    /// Coordinate change x = x' + r, y = y' + s x' + t (i.e. the standard
    /// (u, r, s, t) = (1, r, s, t) transformation; Silverman AEC III Table 3.1).
    fn rst(&self, r: &Integer, s: &Integer, t: &Integer) -> WEq {
        let two = Integer::from(2);
        let three = Integer::from(3);
        let a1 = self.a1.clone() + two.clone() * s.clone();
        let a2 = self.a2.clone() - s.clone() * self.a1.clone() + three.clone() * r.clone()
            - s.clone() * s.clone();
        let a3 = self.a3.clone() + r.clone() * self.a1.clone() + two.clone() * t.clone();
        let a4 = self.a4.clone() - s.clone() * self.a3.clone()
            + two * r.clone() * self.a2.clone()
            - (t.clone() + r.clone() * s.clone()) * self.a1.clone()
            + three * r.clone() * r.clone()
            - Integer::from(2) * s.clone() * t.clone();
        let a6 = self.a6.clone() + r.clone() * self.a4.clone()
            + r.clone() * r.clone() * self.a2.clone()
            + r.clone() * r.clone() * r.clone()
            - t.clone() * self.a3.clone()
            - t.clone() * t.clone()
            - r.clone() * t.clone() * self.a1.clone();
        WEq { a1, a2, a3, a4, a6 }
    }

    /// Rescale by u = p: replaces a_i by a_i / p^i. All divisions must be
    /// exact (checked); used only when the model has been proven
    /// non-minimal at p.
    fn scale_down(&self, p: &Integer) -> WEq {
        WEq {
            a1: exact_div(&self.a1, p),
            a2: exact_div(&self.a2, &p.pow(2)),
            a3: exact_div(&self.a3, &p.pow(3)),
            a4: exact_div(&self.a4, &p.pow(4)),
            a6: exact_div(&self.a6, &p.pow(6)),
        }
    }
}

/// v_p(x); u32::MAX (treated as +infinity) when x = 0.
fn vp(x: &Integer, p: &Integer) -> u32 {
    x.valuation(p)
}

fn pdiv(x: &Integer, p: &Integer) -> bool {
    (x.clone() % p.clone()).is_zero()
}

fn exact_div(x: &Integer, d: &Integer) -> Integer {
    debug_assert!(pdiv(x, d), "exact_div: {} does not divide {}", d, x);
    x.clone() / d.clone()
}

/// Legendre symbol test for a nonzero residue: is `a` a nonzero square mod
/// the odd prime p? The caller must guarantee p is an odd prime and
/// a != 0 mod p (asserted).
///
/// The input is reduced to a non-negative residue before calling
/// `Integer::legendre_symbol`, which mishandles negative inputs (its
/// internal `mod_pow` can return a negative representative that is then
/// compared against p-1; see the report accompanying this module).
fn is_nonzero_square_mod(a: &Integer, p: &Integer) -> bool {
    let r = a.modulo(p);
    assert!(
        !r.is_zero(),
        "is_nonzero_square_mod requires a nonzero residue"
    );
    let sym = r
        .legendre_symbol(p)
        .expect("p is an odd prime by construction");
    assert!(sym != 0, "legendre symbol of nonzero residue cannot be 0");
    sym == 1
}

/// Do the roots of the quadratic Y^2 + a Y - b, which is known to have
/// distinct roots over the algebraic closure of F_p (i.e. p does not divide
/// its discriminant a^2 + 4b), lie in F_p?
fn quadratic_roots_in_fp(a: &Integer, b: &Integer, p: &Integer) -> bool {
    let two = Integer::from(2);
    if *p == two {
        // Evaluate at Y = 0 and Y = 1.
        pdiv(&-b.clone(), p) || pdiv(&(Integer::from(1) + a.clone() - b.clone()), p)
    } else {
        let disc = a.clone() * a.clone() + Integer::from(4) * b.clone();
        is_nonzero_square_mod(&disc, p)
    }
}

/// Do the roots of a X^2 + b X + c, known to have p not dividing its
/// discriminant b^2 - 4ac (distinct roots) and p not dividing a, lie in F_p?
fn general_quadratic_roots_in_fp(a: &Integer, b: &Integer, c: &Integer, p: &Integer) -> bool {
    let two = Integer::from(2);
    if *p == two {
        // Evaluate at X = 0 and X = 1.
        pdiv(c, p) || pdiv(&(a.clone() + b.clone() + c.clone()), p)
    } else {
        let disc = b.clone() * b.clone() - Integer::from(4) * a.clone() * c.clone();
        is_nonzero_square_mod(&disc, p)
    }
}

// ---------------------------------------------------------------------------
// Root counting for cubics over F_p (used for the I0* Tamagawa number).
// ---------------------------------------------------------------------------

fn poly_trim(v: &mut Vec<Integer>) {
    while v.last().is_some_and(|c| c.is_zero()) {
        v.pop();
    }
}

/// Multiply two polynomials of degree <= 2 in F_p[T]/(f), f a monic cubic.
fn poly_mul_mod(a: &[Integer], b: &[Integer], f: &[Integer; 4], p: &Integer) -> Vec<Integer> {
    let mut prod = vec![Integer::zero(); 5];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            prod[i + j] = (prod[i + j].clone() + ai.clone() * bj.clone()).modulo(p);
        }
    }
    // Reduce modulo the monic cubic f.
    for k in (3..prod.len()).rev() {
        let q = prod[k].clone();
        if !q.is_zero() {
            for (j, fj) in f.iter().enumerate().take(3) {
                prod[k - 3 + j] = (prod[k - 3 + j].clone() - q.clone() * fj.clone()).modulo(p);
            }
        }
        prod[k] = Integer::zero();
    }
    prod.truncate(3);
    poly_trim(&mut prod);
    prod
}

/// Degree of gcd(a, b) in F_p[T]; inputs low-to-high, reduced mod p.
fn poly_gcd_degree(mut a: Vec<Integer>, mut b: Vec<Integer>, p: &Integer) -> usize {
    poly_trim(&mut a);
    poly_trim(&mut b);
    loop {
        if b.is_empty() {
            return a.len().saturating_sub(1);
        }
        // a <- a mod b
        let inv = b
            .last()
            .unwrap()
            .mod_inverse(p)
            .expect("nonzero leading coefficient is invertible mod a prime");
        while a.len() >= b.len() && !a.is_empty() {
            let shift = a.len() - b.len();
            let q = (a.last().unwrap().clone() * inv.clone()).modulo(p);
            for (j, bj) in b.iter().enumerate() {
                a[shift + j] = (a[shift + j].clone() - q.clone() * bj.clone()).modulo(p);
            }
            poly_trim(&mut a);
        }
        std::mem::swap(&mut a, &mut b);
    }
}

/// Number of distinct roots in F_p of the monic cubic T^3 + b T^2 + c T + d,
/// computed as deg gcd(T^p - T, f) via square-and-multiply for the Frobenius
/// power. Works for every prime p (no brute force over the residue field).
fn cubic_distinct_roots_in_fp(b: &Integer, c: &Integer, d: &Integer, p: &Integer) -> u32 {
    let f = [d.modulo(p), c.modulo(p), b.modulo(p), Integer::one()];
    // T^p mod (f, p)
    let mut acc = vec![Integer::one()];
    let mut base = vec![Integer::zero(), Integer::one()];
    let mut e = p.clone();
    let two = Integer::from(2);
    while !e.is_zero() {
        if e.is_odd() {
            acc = poly_mul_mod(&acc, &base, &f, p);
        }
        base = poly_mul_mod(&base, &base, &f, p);
        e = e / two.clone();
    }
    // g = T^p - T
    let mut g = acc;
    while g.len() < 2 {
        g.push(Integer::zero());
    }
    g[1] = (g[1].clone() - Integer::one()).modulo(p);
    poly_trim(&mut g);
    let fvec = f.to_vec();
    poly_gcd_degree(fvec, g, p) as u32
}

// ---------------------------------------------------------------------------
// The algorithm proper.
// ---------------------------------------------------------------------------

/// Run Tate's algorithm on `curve` at the prime `p`, returning the local
/// reduction data (Kodaira type, conductor exponent, Tamagawa number,
/// p-minimal model).
///
/// # Panics
///
/// Panics if `p` is not a prime or if the curve is singular
/// (discriminant zero).
pub fn tate_local_data(curve: &EllipticCurve, p: &Integer) -> LocalData {
    assert!(is_prime(p), "tate_local_data: p = {} is not prime", p);
    assert!(
        !curve.is_singular(),
        "tate_local_data: curve is singular (discriminant 0)"
    );

    let two = Integer::from(2);
    let three = Integer::from(3);
    let p_is_2 = *p == two;
    let p_is_3 = *p == three;
    // Inverse of 2 mod p (odd p only).
    let inv2 = if p_is_2 {
        Integer::zero() // never used
    } else {
        two.mod_inverse(p).expect("2 invertible mod odd prime")
    };

    let mut e = WEq {
        a1: curve.a1.clone(),
        a2: curve.a2.clone(),
        a3: curve.a3.clone(),
        a4: curve.a4.clone(),
        a6: curve.a6.clone(),
    };

    // The outer loop restarts (with v_p(disc) reduced by 12) each time the
    // model is found to be non-minimal at p; it must terminate.
    let max_restarts = vp(&e.disc(), p) / 12 + 2;
    let mut restarts = 0u32;

    let (kodaira, f_exp, tamagawa, reduction) = 'restart: loop {
        restarts += 1;
        assert!(
            restarts <= max_restarts,
            "Tate's algorithm failed to terminate: this is a bug"
        );

        let disc = e.disc();
        let n = vp(&disc, p);

        // Step 1: good reduction.
        if n == 0 {
            break (KodairaSymbol::In(0), 0u32, 1u32, ReductionType::Good);
        }

        // Step 2: change coordinates so that the singular point of the
        // reduced curve is (0, 0), i.e. p | a3, a4, a6.
        {
            let (r, t) = if p_is_2 {
                if pdiv(&e.b2(), p) {
                    // a1 even: x0 = sqrt(a4), y0 = sqrt(x0^3 + a2 x0^2 + a4 x0 + a6)
                    let r = e.a4.modulo(p);
                    let y2 = ((r.clone() + e.a2.clone()) * r.clone() + e.a4.clone()) * r.clone()
                        + e.a6.clone();
                    (r, y2.modulo(p))
                } else {
                    // a1 odd: x0 = a3/a1, y0 = (a4 + x0^2)/a1
                    let r = e.a3.modulo(p);
                    let t = (e.a4.clone() + r.clone() * r.clone()).modulo(p);
                    (r, t)
                }
            } else if p_is_3 {
                let r = if pdiv(&e.b2(), p) {
                    // cube root of -b6 mod 3 is -b6 itself
                    (-e.b6()).modulo(p)
                } else {
                    let invb2 = e.b2().modulo(p).mod_inverse(p).expect("b2 unit mod 3");
                    (-(e.b4()) * invb2).modulo(p)
                };
                let t = (e.a1.clone() * r.clone() + e.a3.clone()).modulo(p);
                (r, t)
            } else {
                let c4 = e.c4();
                let inv12 = Integer::from(12)
                    .mod_inverse(p)
                    .expect("12 invertible mod p >= 5");
                let r = if pdiv(&c4, p) {
                    // cusp: triple root of the RHS cubic, x0 = -b2/12
                    (-e.b2() * inv12).modulo(p)
                } else {
                    // node: double root, x0 = -(c6 + b2 c4)/(12 c4)
                    let inv12c4 = (Integer::from(12) * c4.clone())
                        .modulo(p)
                        .mod_inverse(p)
                        .expect("12*c4 unit mod p here");
                    (-(e.c6() + e.b2() * c4) * inv12c4).modulo(p)
                };
                let t = (-(e.a1.clone() * r.clone() + e.a3.clone()) * inv2.clone()).modulo(p);
                (r, t)
            };
            e = e.rst(&r, &Integer::zero(), &t);
        }
        assert!(
            pdiv(&e.a3, p) && pdiv(&e.a4, p) && pdiv(&e.a6, p),
            "Tate step 2 postcondition failed: this is a bug"
        );

        // Step 3: multiplicative reduction (node): p does not divide b2
        // (equivalent to p not dividing c4 once the singular point is at the
        // origin).
        let b2 = e.b2();
        if !pdiv(&b2, p) {
            // Type I_n, n = v_p(disc). Split iff the tangent-cone quadratic
            // T^2 + a1 T - a2 splits over F_p: for odd p its discriminant is
            // b2, so split iff b2 is a square mod p; for p = 2 (where a1 is
            // necessarily odd) split iff a2 is even.
            let split = if p_is_2 {
                pdiv(&e.a2, p)
            } else {
                is_nonzero_square_mod(&b2, p)
            };
            let c = if split {
                n
            } else if n.is_multiple_of(2) {
                2
            } else {
                1
            };
            let red = if split {
                ReductionType::SplitMultiplicative
            } else {
                ReductionType::NonsplitMultiplicative
            };
            break (KodairaSymbol::In(n), 1, c, red);
        }

        // Additive reduction from here on.

        // Step 4: type II iff p^2 does not divide a6.
        if vp(&e.a6, p) < 2 {
            break (KodairaSymbol::II, n, 1, ReductionType::Additive);
        }

        // Step 5: type III iff p^3 does not divide b8.
        if vp(&e.b8(), p) < 3 {
            break (KodairaSymbol::III, n - 1, 2, ReductionType::Additive);
        }

        // Step 6: type IV iff p^3 does not divide b6. Tamagawa number is 3
        // if Y^2 + (a3/p) Y - (a6/p^2) has roots in F_p, else 1.
        if vp(&e.b6(), p) < 3 {
            let a3t = exact_div(&e.a3, p);
            let a6t = exact_div(&e.a6, &p.pow(2));
            let c = if quadratic_roots_in_fp(&a3t, &a6t, p) {
                3
            } else {
                1
            };
            break (KodairaSymbol::IV, n - 2, c, ReductionType::Additive);
        }

        // Step 7 preparation: change coordinates so that
        // p | a1, a2, p^2 | a3, a4 and p^3 | a6.
        {
            let (s, t) = if p_is_2 {
                let s = e.a2.modulo(p);
                // v(a6) >= 2 here (we are past type II)
                let t = p.clone() * exact_div(&e.a6, &p.pow(2)).modulo(p);
                (s, t)
            } else {
                // halfmodp trick: with h = (p+1)/2 an exact integer,
                // a1 + 2*(-a1*h) = -p*a1 and a3 + 2*(-a3*h) = -p*a3, so the
                // divisibility gained is exact, not merely mod p.
                let h = (p.clone() + Integer::one()) / two.clone();
                let s = -e.a1.clone() * h.clone();
                let t = -e.a3.clone() * h;
                (s, t)
            };
            e = e.rst(&Integer::zero(), &s, &t);
        }
        assert!(
            pdiv(&e.a1, p)
                && pdiv(&e.a2, p)
                && vp(&e.a3, p) >= 2
                && vp(&e.a4, p) >= 2
                && vp(&e.a6, p) >= 3,
            "Tate step 7 preparation postcondition failed: this is a bug"
        );

        // The reduced equation is now Y^2 = cubic in X; analyse the cubic
        // T^3 + b T^2 + c T + d mod p, where b = a2/p, c = a4/p^2, d = a6/p^3.
        let cb = exact_div(&e.a2, p);
        let cc = exact_div(&e.a4, &p.pow(2));
        let cd = exact_div(&e.a6, &p.pow(3));
        // w = -disc(cubic); multiple root iff p | w.
        let w = Integer::from(27) * cd.clone() * cd.clone()
            - cb.clone() * cb.clone() * cc.clone() * cc.clone()
            + Integer::from(4) * cb.clone() * cb.clone() * cb.clone() * cd.clone()
            - Integer::from(18) * cb.clone() * cc.clone() * cd.clone()
            + Integer::from(4) * cc.clone() * cc.clone() * cc.clone();
        // x != 0 mod p distinguishes a double root from a triple root.
        let xq = Integer::from(3) * cc.clone() - cb.clone() * cb.clone();

        if !pdiv(&w, p) {
            // Step 7 (distinct roots): type I0*.
            // c_p = 1 + #{roots of the cubic in F_p} (1, 2 or 4).
            let c = 1 + cubic_distinct_roots_in_fp(&cb, &cc, &cd, p);
            assert!(c == 1 || c == 2 || c == 4);
            break (KodairaSymbol::InStar(0), n - 4, c, ReductionType::Additive);
        }

        if !pdiv(&xq, p) {
            // Step 8 (double root): type I_m* for some m >= 1.
            // Move the double root of the cubic to T = 0.
            let r1 = if p_is_2 {
                cc.modulo(p)
            } else if p_is_3 {
                (cb.clone() * cc.clone()).modulo(p)
            } else {
                let inv2x = (Integer::from(2) * xq.clone())
                    .modulo(p)
                    .mod_inverse(p)
                    .expect("2x unit mod p here");
                ((cb.clone() * cc.clone() - Integer::from(9) * cd.clone()) * inv2x).modulo(p)
            };
            let r = p.clone() * r1;
            e = e.rst(&r, &Integer::zero(), &Integer::zero());
            assert!(
                vp(&e.a2, p) == 1 && vp(&e.a3, p) >= 2 && vp(&e.a4, p) >= 3 && vp(&e.a6, p) >= 4,
                "Tate step 8 postcondition failed: this is a bug"
            );

            // Sub-loop: examine alternately a quadratic in Y and one in X,
            // translating away double roots, until one has distinct roots.
            let mut ix: u32 = 3;
            let mut iy: u32 = 3;
            let mut mx = p.pow(2);
            let mut my = mx.clone();
            let mut guard = 0u32;
            let cp = loop {
                guard += 1;
                assert!(guard <= n + 4, "I_m* sub-loop failed to terminate: bug");

                let a3t = exact_div(&e.a3, &my);
                let a6t = exact_div(&e.a6, &(mx.clone() * my.clone()));
                // Quadratic in Y: Y^2 + a3t Y - a6t.
                let disc_y = a3t.clone() * a3t.clone() + Integer::from(4) * a6t.clone();
                if !pdiv(&disc_y, p) {
                    break if quadratic_roots_in_fp(&a3t, &a6t, p) {
                        4
                    } else {
                        2
                    };
                }
                // Double root: translate it to Y = 0.
                let t1 = if p_is_2 {
                    a6t.modulo(p)
                } else {
                    (-a3t * inv2.clone()).modulo(p)
                };
                e = e.rst(&Integer::zero(), &Integer::zero(), &(my.clone() * t1));
                my = my.clone() * p.clone();
                iy += 1;

                let a2t = exact_div(&e.a2, p);
                let a4t = exact_div(&e.a4, &(p.clone() * mx.clone()));
                let a6t = exact_div(&e.a6, &(mx.clone() * my.clone()));
                // Quadratic in X: a2t X^2 + a4t X + a6t.
                let disc_x = a4t.clone() * a4t.clone()
                    - Integer::from(4) * a2t.clone() * a6t.clone();
                if !pdiv(&disc_x, p) {
                    break if general_quadratic_roots_in_fp(&a2t, &a4t, &a6t, p) {
                        4
                    } else {
                        2
                    };
                }
                // Double root: translate it to X = 0.
                let r1 = if p_is_2 {
                    let inva2t = a2t.modulo(p).mod_inverse(p).expect("a2t unit");
                    (a6t * inva2t).modulo(p)
                } else {
                    let inv2a2t = (Integer::from(2) * a2t)
                        .modulo(p)
                        .mod_inverse(p)
                        .expect("2*a2t unit mod p here");
                    (-a4t * inv2a2t).modulo(p)
                };
                e = e.rst(&(mx.clone() * r1), &Integer::zero(), &Integer::zero());
                mx = mx.clone() * p.clone();
                ix += 1;
            };
            let m = ix + iy - 5;
            assert!(m >= 1 && n >= 4 + m);
            break (
                KodairaSymbol::InStar(m),
                n - 4 - m,
                cp,
                ReductionType::Additive,
            );
        }

        // Step 9 (triple root): move it to T = 0.
        {
            let r1 = if p_is_2 {
                cb.modulo(p)
            } else if p_is_3 {
                (-cd.clone()).modulo(p)
            } else {
                let inv3 = three.mod_inverse(p).expect("3 invertible mod p >= 5");
                (-cb.clone() * inv3).modulo(p)
            };
            let r = p.clone() * r1;
            e = e.rst(&r, &Integer::zero(), &Integer::zero());
        }
        assert!(
            vp(&e.a2, p) >= 2 && vp(&e.a3, p) >= 2 && vp(&e.a4, p) >= 3 && vp(&e.a6, p) >= 4,
            "Tate step 9 postcondition failed: this is a bug"
        );

        // Type IV* iff the quadratic Y^2 + (a3/p^2) Y - (a6/p^4) has
        // distinct roots mod p; c_p = 3 if they are rational, else 1.
        let a3t = exact_div(&e.a3, &p.pow(2));
        let a6t = exact_div(&e.a6, &p.pow(4));
        let disc_y = a3t.clone() * a3t.clone() + Integer::from(4) * a6t.clone();
        if !pdiv(&disc_y, p) {
            let c = if quadratic_roots_in_fp(&a3t, &a6t, p) {
                3
            } else {
                1
            };
            break (KodairaSymbol::IVStar, n - 6, c, ReductionType::Additive);
        }

        // Double root: translate it to Y = 0 (so p^3 | a3, p^5 | a6).
        {
            let t1 = if p_is_2 {
                a6t.modulo(p)
            } else {
                (-a3t * inv2.clone()).modulo(p)
            };
            let t = p.pow(2) * t1;
            e = e.rst(&Integer::zero(), &Integer::zero(), &t);
        }
        assert!(
            vp(&e.a3, p) >= 3 && vp(&e.a6, p) >= 5,
            "Tate step 10 precondition failed: this is a bug"
        );

        // Step 10: type III* iff p^4 does not divide a4.
        if vp(&e.a4, p) < 4 {
            break (KodairaSymbol::IIIStar, n - 7, 2, ReductionType::Additive);
        }

        // Type II* iff p^6 does not divide a6.
        if vp(&e.a6, p) < 6 {
            break (KodairaSymbol::IIStar, n - 8, 1, ReductionType::Additive);
        }

        // Step 11: the equation is not minimal at p; rescale by u = p and
        // restart. (v(a1) >= 1, v(a2) >= 2, v(a3) >= 3, v(a4) >= 4,
        // v(a6) >= 6 all hold here.)
        assert!(vp(&e.a1, p) >= 1 && vp(&e.a2, p) >= 2, "rescale precondition");
        e = e.scale_down(p);
        continue 'restart;
    };

    let minimal_model = EllipticCurve::new(
        e.a1.clone(),
        e.a2.clone(),
        e.a3.clone(),
        e.a4.clone(),
        e.a6.clone(),
    );
    let min_disc_val = vp(&minimal_model.discriminant, p);

    // Independent cross-checks (these are theorems about the output of
    // Tate's algorithm; violating any of them means an implementation bug).
    if *p >= Integer::from(5) && reduction == ReductionType::Additive {
        assert!(f_exp == 2, "tame additive reduction must have f_p = 2");
    }
    if p_is_3 {
        assert!(f_exp <= 5, "f_3 <= 5 always");
    }
    if p_is_2 {
        assert!(f_exp <= 8, "f_2 <= 8 always");
    }

    LocalData {
        prime: p.clone(),
        kodaira,
        conductor_exponent: f_exp,
        tamagawa_number: tamagawa,
        minimal_model,
        minimal_disc_valuation: min_disc_val,
        reduction,
    }
}

/// The conductor N = prod_p p^{f_p} of the curve, with each local exponent
/// f_p computed by Tate's algorithm at every prime dividing the
/// discriminant of the given model. (Primes at which the given model is
/// non-minimal but the curve has good reduction contribute f_p = 0.)
///
/// Cost note: this factors |disc| with the trial-division based
/// `rustmath_integers::prime::factor`, so it is only practical when the
/// discriminant has no huge prime-square obstructions to trial division;
/// fine for the moderate discriminants this crate currently targets.
///
/// # Panics
///
/// Panics if the curve is singular.
pub fn conductor(curve: &EllipticCurve) -> Integer {
    assert!(
        !curve.is_singular(),
        "conductor: curve is singular (discriminant 0)"
    );
    let mut n = Integer::one();
    for (p, _) in factor(&curve.discriminant.abs()) {
        let ld = tate_local_data(curve, &p);
        n = n * p.pow(ld.conductor_exponent);
    }
    n
}

impl EllipticCurve {
    /// Local reduction data at the prime p (Tate's algorithm): Kodaira
    /// type, conductor exponent f_p, Tamagawa number c_p, p-minimal model.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not prime or the curve is singular.
    pub fn local_data(&self, p: &Integer) -> LocalData {
        tate_local_data(self, p)
    }

    /// The reduction type at the prime p (good / split multiplicative /
    /// non-split multiplicative / additive), from Tate's algorithm.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not prime or the curve is singular.
    pub fn reduction_type(&self, p: &Integer) -> ReductionType {
        self.local_data(p).reduction
    }

    /// The conductor of the curve, N = prod p^{f_p}, with every exponent
    /// computed by Tate's algorithm (this is the true conductor, not the
    /// squarefree "product of bad primes" semistable approximation).
    ///
    /// # Panics
    ///
    /// Panics if the curve is singular.
    pub fn compute_conductor(&self) -> Integer {
        conductor(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(a1: i64, a2: i64, a3: i64, a4: i64, a6: i64) -> EllipticCurve {
        EllipticCurve::new(
            Integer::from(a1),
            Integer::from(a2),
            Integer::from(a3),
            Integer::from(a4),
            Integer::from(a6),
        )
    }

    /// Assert the full local datum. `kod` uses the Display form ("I5",
    /// "IV*", ...).
    fn check_local(e: &EllipticCurve, p: i64, kod: &str, f: u32, c: u32, vdisc: u32) {
        let ld = e.local_data(&Integer::from(p));
        assert_eq!(ld.kodaira.to_string(), kod, "Kodaira type at p={}", p);
        assert_eq!(ld.conductor_exponent, f, "conductor exponent at p={}", p);
        assert_eq!(ld.tamagawa_number, c, "Tamagawa number at p={}", p);
        assert_eq!(
            ld.minimal_disc_valuation, vdisc,
            "minimal disc valuation at p={}",
            p
        );
        // The p-minimal model must be an integral model of the same curve up
        // to the transformations used, in particular nonsingular with the
        // predicted discriminant valuation.
        assert!(!ld.minimal_model.is_singular());
    }

    // Ground truth for every expected value in the tests below was
    // independently verified with PARI/GP 2.x on this machine:
    //   elllocalred(ellinit([a1,a2,a3,a4,a6]), p) -> [f, kod, [u,r,s,t], c]
    //   ellglobalred(...) -> N,  ellap(...,p) for split/non-split.
    // Curve labels are from the Cremona tables.

    #[test]
    fn test_11a1_multiplicative_split() {
        // 11a1: y^2 + y = x^3 - x^2 - 10x - 20, disc = -11^5, N = 11.
        // PARI: elllocalred at 11 -> f=1, I5, c=5; ellap(E,11) = 1 (split).
        let e = curve(0, -1, 1, -10, -20);
        assert_eq!(e.discriminant, Integer::from(-161051)); // -11^5
        check_local(&e, 11, "I5", 1, 5, 5);
        assert_eq!(
            e.reduction_type(&Integer::from(11)),
            ReductionType::SplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(11));
    }

    #[test]
    fn test_37a1_multiplicative_nonsplit() {
        // 37a1: y^2 + y = x^3 - x, disc = 37, N = 37 (rank 1 curve).
        // PARI: at 37 -> f=1, I1, c=1; ellap(E,37) = -1 (non-split).
        let e = curve(0, 0, 1, -1, 0);
        assert_eq!(e.discriminant, Integer::from(37));
        check_local(&e, 37, "I1", 1, 1, 1);
        assert_eq!(
            e.reduction_type(&Integer::from(37)),
            ReductionType::NonsplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(37));
    }

    #[test]
    fn test_389a1() {
        // 389a1: y^2 + y = x^3 + x^2 - 2x, disc = 389, N = 389 (rank 2).
        // PARI: at 389 -> f=1, I1, c=1; ellap = 1 (split).
        let e = curve(0, 1, 1, -2, 0);
        assert_eq!(e.discriminant, Integer::from(389));
        check_local(&e, 389, "I1", 1, 1, 1);
        assert_eq!(
            e.reduction_type(&Integer::from(389)),
            ReductionType::SplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(389));
    }

    #[test]
    fn test_15a1_two_bad_primes() {
        // 15a1: y^2 + xy + y = x^3 + x^2 - 10x - 10, disc = 50625 = 3^4 5^4,
        // N = 15. PARI: at 3 -> I4, c=2 (ellap=-1, non-split, n even);
        // at 5 -> I4, c=4 (ellap=1, split).
        let e = curve(1, 1, 1, -10, -10);
        assert_eq!(e.discriminant, Integer::from(50625));
        check_local(&e, 3, "I4", 1, 2, 4);
        check_local(&e, 5, "I4", 1, 4, 4);
        assert_eq!(
            e.reduction_type(&Integer::from(3)),
            ReductionType::NonsplitMultiplicative
        );
        assert_eq!(
            e.reduction_type(&Integer::from(5)),
            ReductionType::SplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(15));
    }

    #[test]
    fn test_49a1_additive() {
        // 49a1: y^2 + xy = x^3 - x^2 - 2x - 1, disc = -343 = -7^3, N = 49.
        // PARI: at 7 -> f=2, III, c=2.
        let e = curve(1, -1, 0, -2, -1);
        assert_eq!(e.discriminant, Integer::from(-343));
        check_local(&e, 7, "III", 2, 2, 3);
        assert_eq!(
            e.reduction_type(&Integer::from(7)),
            ReductionType::Additive
        );
        assert_eq!(e.compute_conductor(), Integer::from(49));
    }

    #[test]
    fn test_27a1_wild_at_3() {
        // 27a1: y^2 + y = x^3 - 7, disc = -3^9, N = 27 (wild at 3).
        // PARI: at 3 -> f=3, IV*, c=3. (Also hand-walked through the
        // algorithm in the development notes.)
        let e = curve(0, 0, 1, 0, -7);
        assert_eq!(e.discriminant, Integer::from(-19683));
        check_local(&e, 3, "IV*", 3, 3, 9);
        assert_eq!(e.compute_conductor(), Integer::from(27));
    }

    #[test]
    fn test_32a1_wild_at_2() {
        // 32a1: y^2 = x^3 - x, disc = 64, N = 32 (wild at 2).
        // PARI: at 2 -> f=5, III, c=2.
        let e = curve(0, 0, 0, -1, 0);
        assert_eq!(e.discriminant, Integer::from(64));
        check_local(&e, 2, "III", 5, 2, 6);
        assert_eq!(e.compute_conductor(), Integer::from(32));
    }

    #[test]
    fn test_36a1_two_additive_primes() {
        // 36a1: y^2 = x^3 + 1, disc = -432, N = 36.
        // PARI: at 2 -> f=2, IV, c=3; at 3 -> f=2, III, c=2.
        let e = curve(0, 0, 0, 0, 1);
        assert_eq!(e.discriminant, Integer::from(-432));
        check_local(&e, 2, "IV", 2, 3, 4);
        check_local(&e, 3, "III", 2, 2, 3);
        assert_eq!(e.compute_conductor(), Integer::from(36));
    }

    #[test]
    fn test_14a1_nonsplit_at_2() {
        // 14a1: y^2 + xy + y = x^3 + 4x - 6, disc = -21952, N = 14.
        // PARI: at 2 -> I6, c=2 (ellap=-1, non-split, n even);
        // at 7 -> I3, c=3 (ellap=1, split).
        let e = curve(1, 0, 1, 4, -6);
        check_local(&e, 2, "I6", 1, 2, 6);
        check_local(&e, 7, "I3", 1, 3, 3);
        assert_eq!(
            e.reduction_type(&Integer::from(2)),
            ReductionType::NonsplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(14));
    }

    #[test]
    fn test_split_multiplicative_at_2() {
        // [1,-2,0,-3,1]: disc = 3592 = 2^3 * 449, N = 898.
        // PARI: at 2 -> I3, c=3, ellap=1 (split, so c = n = 3; a non-split
        // I3 would have c = 1 -- this pins the split test at p = 2).
        let e = curve(1, -2, 0, -3, 1);
        assert_eq!(e.discriminant, Integer::from(3592));
        check_local(&e, 2, "I3", 1, 3, 3);
        assert_eq!(
            e.reduction_type(&Integer::from(2)),
            ReductionType::SplitMultiplicative
        );
        check_local(&e, 449, "I1", 1, 1, 1);
        assert_eq!(e.compute_conductor(), Integer::from(898));
    }

    #[test]
    fn test_i0star() {
        // y^2 = x^3 - 25x (congruent-number curve for 5), N = 800.
        // PARI: at 5 -> f=2, I0*, c=4 (the cubic T^3 - T has all three
        // roots in F_5); at 2 -> f=5, III, c=2.
        let e = curve(0, 0, 0, -25, 0);
        check_local(&e, 5, "I0*", 2, 4, 6);
        check_local(&e, 2, "III", 5, 2, 6);
        assert_eq!(e.compute_conductor(), Integer::from(800));
    }

    #[test]
    fn test_i0star_c2() {
        // y^2 = x^3 + 25x + 250: PARI: at 5 -> I0*, c=2 (cubic T^3+T+2 has
        // exactly one root in F_5); at 2 -> I1*, c=2 (a wild I_m* at 2 with
        // NON-rational roots, pinning the c=2 branch); at 7 -> I1 c=1.
        // N = 1400.
        let e = curve(0, 0, 0, 25, 250);
        check_local(&e, 5, "I0*", 2, 2, 6);
        check_local(&e, 2, "I1*", 3, 2, 8);
        check_local(&e, 7, "I1", 1, 1, 1);
        assert_eq!(e.compute_conductor(), Integer::from(1400));
    }

    #[test]
    fn test_i1star() {
        // y^2 = x^3 - 5x^2 + 625: PARI: at 5 -> f=2, I1*, c=4.
        let e = curve(0, -5, 0, 0, 625);
        check_local(&e, 5, "I1*", 2, 4, 7);
    }

    #[test]
    fn test_i2star() {
        // y^2 = x^3 + 50x + 375: PARI: at 5 -> f=2, I2*, c=4;
        // at 2 -> f=4, II, c=1 (wild II at 2); at 11 -> I1, c=1. N = 4400.
        let e = curve(0, 0, 0, 50, 375);
        check_local(&e, 5, "I2*", 2, 4, 8);
        check_local(&e, 2, "II", 4, 1, 4);
        check_local(&e, 11, "I1", 1, 1, 1);
        assert_eq!(e.compute_conductor(), Integer::from(4400));
    }

    #[test]
    fn test_imstar_at_2_from_cremona_24a() {
        // [0,-1,0,-4,4] (Cremona 24a1): N = 24.
        // PARI: at 2 -> f=3, I1*, c=4; at 3 -> I2, c=2 (non-split).
        let e = curve(0, -1, 0, -4, 4);
        check_local(&e, 2, "I1*", 3, 4, 8);
        check_local(&e, 3, "I2", 1, 2, 2);
        assert_eq!(e.compute_conductor(), Integer::from(24));
    }

    #[test]
    fn test_i0star_at_2_from_cremona_48a() {
        // [0,1,0,-4,-4] (Cremona 48a1): N = 48.
        // PARI: at 2 -> f=4, I0*, c=2; at 3 -> I2, c=2 (split, n = c = 2).
        let e = curve(0, 1, 0, -4, -4);
        check_local(&e, 2, "I0*", 4, 2, 8);
        check_local(&e, 3, "I2", 1, 2, 2);
        assert_eq!(
            e.reduction_type(&Integer::from(3)),
            ReductionType::SplitMultiplicative
        );
        assert_eq!(e.compute_conductor(), Integer::from(48));
    }

    #[test]
    fn test_type_ii_and_iv_ladder_at_5() {
        // y^2 = x^3 + 5^k for k = 1..5 walks the additive ladder at p=5:
        // k=1: II c=1; k=2: IV c=3; k=4: IV* c=3; k=5: II* c=1.
        // All PARI-verified (f=2 at 5 in each case, tame).
        check_local(&curve(0, 0, 0, 0, 5), 5, "II", 2, 1, 2);
        check_local(&curve(0, 0, 0, 0, 25), 5, "IV", 2, 3, 4);
        check_local(&curve(0, 0, 0, 0, 625), 5, "IV*", 2, 3, 8);
        check_local(&curve(0, 0, 0, 0, 3125), 5, "II*", 2, 1, 10);
        // y^2 = x^3 + 125x: PARI: at 5 -> III*, c=2.
        check_local(&curve(0, 0, 0, 125, 0), 5, "III*", 2, 2, 9);
        // y^2 = x^3 + 50: PARI: at 5 -> IV with NON-rational roots, c=1.
        check_local(&curve(0, 0, 0, 0, 50), 5, "IV", 2, 1, 4);
    }

    #[test]
    fn test_wild_types_at_2_and_3() {
        // y^2 = x^3 + 5: PARI: at 3 -> f=3, II, c=1 (wild II at 3);
        // at 2 -> f=2, IV, c=1 (IV at 2 with non-rational roots).
        let e = curve(0, 0, 0, 0, 5);
        check_local(&e, 3, "II", 3, 1, 3);
        check_local(&e, 2, "IV", 2, 1, 4);
        // y^2 = x^3 + 9x^2 + 81: PARI: at 3 -> f=4, IV*, c=3 (wild IV* at 3).
        let e = curve(0, 9, 0, 0, 81);
        check_local(&e, 3, "IV*", 4, 3, 10);
        // y^2 = x^3 + 125x: PARI: at 2 -> f=6, II, c=1 (wild II at 2, f=6).
        let e = curve(0, 0, 0, 125, 0);
        check_local(&e, 2, "II", 6, 1, 6);
        // y^2 = x^3 + x: PARI: at 2 -> f=6, II, c=1. N = 64.
        let e = curve(0, 0, 0, 1, 0);
        check_local(&e, 2, "II", 6, 1, 6);
        assert_eq!(e.compute_conductor(), Integer::from(64));
    }

    #[test]
    fn test_nonminimal_model_rescaled() {
        // y^2 = x^3 + 5^6 is non-minimal at 5: v_5(disc) = 12, and the
        // algorithm must rescale to y^2 = x^3 + 1 and find GOOD reduction.
        // PARI: at 5 -> f=0, I0, c=1; N = 36.
        let e = curve(0, 0, 0, 0, 15625);
        let ld = e.local_data(&Integer::from(5));
        assert_eq!(ld.kodaira, KodairaSymbol::In(0));
        assert_eq!(ld.conductor_exponent, 0);
        assert_eq!(ld.tamagawa_number, 1);
        assert_eq!(ld.minimal_disc_valuation, 0);
        assert_eq!(ld.reduction, ReductionType::Good);
        // The p-minimal model's discriminant is prime to 5.
        assert!(!pdiv(&ld.minimal_model.discriminant, &Integer::from(5)));
        assert_eq!(e.compute_conductor(), Integer::from(36));
    }

    #[test]
    fn test_nonminimal_scalings_of_11a1() {
        // 11a1 scaled by u = 2, 3, 5: the extra prime gets v(disc) = 12 but
        // good reduction after minimalization; N stays 11. PARI-verified.
        for (u, a) in [
            (2i64, [0i64, -4, 8, -160, -1280]),
            (3, [0, -9, 27, -810, -14580]),
            (5, [0, -25, 125, -6250, -312500]),
        ] {
            let e = curve(a[0], a[1], a[2], a[3], a[4]);
            let ld = e.local_data(&Integer::from(u));
            assert_eq!(ld.kodaira, KodairaSymbol::In(0), "u = {}", u);
            assert_eq!(ld.conductor_exponent, 0);
            check_local(&e, 11, "I5", 1, 5, 5);
            assert_eq!(e.compute_conductor(), Integer::from(11));
        }
    }

    #[test]
    fn test_small_short_weierstrass_conductors() {
        // Conductors of the small curves used elsewhere in this crate's
        // tests, all PARI ellglobalred-verified.
        // y^2 = x^3 - x + 1: N = 92 (IV c=3 at 2, I1 at 23).
        assert_eq!(curve(0, 0, 0, -1, 1).compute_conductor(), Integer::from(92));
        // y^2 = x^3 - 1: N = 144 (II at 2 with f=4, III at 3).
        let e = curve(0, 0, 0, 0, -1);
        check_local(&e, 2, "II", 4, 1, 4);
        check_local(&e, 3, "III", 2, 2, 3);
        assert_eq!(e.compute_conductor(), Integer::from(144));
        // y^2 = x^3 + 2x + 3: N = 880 (II at 2, I2 non-split at 5, I1 at 11).
        let e = curve(0, 0, 0, 2, 3);
        check_local(&e, 2, "II", 4, 1, 4);
        check_local(&e, 5, "I2", 1, 2, 2);
        check_local(&e, 11, "I1", 1, 1, 1);
        assert_eq!(e.compute_conductor(), Integer::from(880));
    }

    #[test]
    fn test_kodaira_display() {
        assert_eq!(KodairaSymbol::In(0).to_string(), "I0");
        assert_eq!(KodairaSymbol::In(5).to_string(), "I5");
        assert_eq!(KodairaSymbol::InStar(0).to_string(), "I0*");
        assert_eq!(KodairaSymbol::InStar(2).to_string(), "I2*");
        assert_eq!(KodairaSymbol::IVStar.to_string(), "IV*");
        assert_eq!(KodairaSymbol::IIStar.to_string(), "II*");
        assert_eq!(KodairaSymbol::IIIStar.to_string(), "III*");
        assert_eq!(KodairaSymbol::II.to_string(), "II");
        assert_eq!(KodairaSymbol::III.to_string(), "III");
        assert_eq!(KodairaSymbol::IV.to_string(), "IV");
    }

    #[test]
    fn test_good_prime_is_i0() {
        // 11a1 has good reduction at every p != 11.
        let e = curve(0, -1, 1, -10, -20);
        for p in [2i64, 3, 5, 7, 13] {
            let ld = e.local_data(&Integer::from(p));
            assert_eq!(ld.kodaira, KodairaSymbol::In(0));
            assert_eq!(ld.conductor_exponent, 0);
            assert_eq!(ld.tamagawa_number, 1);
            assert_eq!(ld.reduction, ReductionType::Good);
        }
    }

    #[test]
    #[should_panic(expected = "not prime")]
    fn test_rejects_composite_p() {
        let e = curve(0, -1, 1, -10, -20);
        let _ = e.local_data(&Integer::from(6));
    }

    #[test]
    #[should_panic(expected = "singular")]
    fn test_rejects_singular_curve() {
        // y^2 = x^3 has discriminant 0.
        let e = curve(0, 0, 0, 0, 0);
        let _ = e.local_data(&Integer::from(2));
    }

    #[test]
    fn test_cubic_root_counter_against_brute_force() {
        // Self-check of the Frobenius-gcd root counter against direct
        // evaluation for a spread of cubics and primes.
        for p in [2i64, 3, 5, 7, 11, 13, 101] {
            let pi = Integer::from(p);
            for (b, c, d) in [
                (0i64, -1i64, 0i64), // T^3 - T: 3 roots for every p > 2
                (0, 0, 1),
                (0, 1, 2),
                (1, 1, 1),
                (0, 0, -1),
                (2, -3, 5),
            ] {
                let expected = (0..p)
                    .filter(|t| {
                        let v = t * t * t + b * t * t + c * t + d;
                        v.rem_euclid(p) == 0
                    })
                    .count() as u32;
                let got = cubic_distinct_roots_in_fp(
                    &Integer::from(b),
                    &Integer::from(c),
                    &Integer::from(d),
                    &pi,
                );
                assert_eq!(got, expected, "cubic ({},{},{}) mod {}", b, c, d, p);
            }
        }
    }
}
