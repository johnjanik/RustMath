//! MAGMA `IsSquare(a)` / `Sqrt(a)` with the root in GF(p) and GF(p^n).
//!
//! MAGMA Handbook ch. 21 §21.4.4 `IsSquare(a)` ("return true and a square root b (with b^2 = a)
//! if a is a square") and §21.4.7 `SquareRoot(a)` / `Sqrt(a)` (digest
//! `rustmath-lava/docs/Chapter_21_Finite_Fields.md`). Algorithm: Euler's criterion decides; the
//! root by a^((q+1)/4) when q = 3 mod 4, by Tonelli-Shanks in the cyclic group GF(q)^* (H.
//! Cohen, *A Course in Computational Algebraic Number Theory*, GTM 138, Alg. 1.5.1), and by the
//! inverse Frobenius a^(q/2) in characteristic 2. The canonical root is the one of r, -r whose
//! coefficient vector read as Σ c_i p^i is smaller.
//!
//! Reference values, PARI/GP 2.15.4 (`gp -q`), with
//! ```text
//! key(a, p) = subst(lift(a.pol), variable(a.pol), p);
//! canon(r, p) = if(key(-r, p) < key(r, p), -r, r);
//! show(a, p) = my(s = issquare(a)); if(s, my(r = canon(sqrt(a), p));
//!   print1("[", Vecrev(lift(a.pol)), " sq ", Vecrev(lift(r.pol)), "] "),
//!   print1("[", Vecrev(lift(a.pol)), " nonsq] "));
//! lr(a, p) = if(!issquare(Mod(a,p)), "nonsq", my(r = lift(sqrt(Mod(a,p)))); min(r, p-r));
//! w = ffgen(Mod(1,3)*(t^2+2*t+2), 't); for(i=0,8, show((i%3) + (i\3)*w, 3))
//! w = ffgen(Mod(1,5)*(t^3+3*t+3), 't); foreach([w, w+1, 2*w^2+3, w^2, 3*w^0, 2*w^0], a, show(a, 5))
//! w = ffgen(Mod(1,2)*(t^8+t^4+t^3+t^2+1), 't); [Vecrev(lift(sqrt(w).pol)), Vecrev(lift(sqrt(w+1).pol))]
//! p = 2^61-1; w = ffgen(Mod(1,p)*(t^2-3), 't);
//!   foreach([w, w+1, w+2, 3*w+5, w^2+7*w, 3*w^0, 12345678901234567*w + 9876543210], a, show(a, p))
//! p = 2^80+13; w = ffgen(Mod(1,p)*(t^2+t+1), 't); foreach([w, w+1, w+2, 3*w+5, 7*w^0, 2*w^0], a, show(a, p))
//! ```
//! The outputs are quoted at each test. `kronecker(3, 2^61-1)` = -1, so t^2 - 3 is irreducible
//! over F_(2^61-1); 2^80+13 = 2 mod 3, so t^2 + t + 1 is irreducible over F_(2^80+13).

use rustmath_core::Ring;
use rustmath_finitefields::{ExtensionField, FiniteField, FiniteFieldElement, PrimeField};
use rustmath_integers::Integer;
use rustmath_polynomials::UnivariatePolynomial;
use std::collections::HashSet;

fn int(n: i64) -> Integer {
    Integer::from(n)
}

fn big(s: &str) -> Integer {
    Integer::from_decimal_str(s).unwrap()
}

fn el(f: &FiniteField, v: &[&str]) -> FiniteFieldElement {
    f.element(v.iter().map(|s| big(s)).collect())
}

/// The coefficient vector of `x`, padded to the field degree.
fn seq(x: &FiniteFieldElement) -> Vec<Integer> {
    let mut v = x.eltseq().to_vec();
    v.resize(x.field().degree(), Integer::zero());
    v
}

fn strs(v: &[&str], n: usize) -> Vec<Integer> {
    let mut out: Vec<Integer> = v.iter().map(|s| big(s)).collect();
    out.resize(n, Integer::zero());
    out
}

/// Checks `sqrt(a)` against the gp row: `Some(root)` or `None` for "nonsq".
fn check(f: &FiniteField, a: &[&str], root: Option<&[&str]>) {
    let x = el(f, a);
    let n = f.degree();
    match root {
        Some(r) => {
            assert!(x.is_square(), "{a:?} is a square");
            let got = x.sqrt().expect("a square");
            assert_eq!(seq(&got), strs(r, n), "sqrt({a:?})");
            assert!(got.clone() * got == x);
        }
        None => {
            assert!(!x.is_square(), "{a:?} is not a square");
            assert!(x.sqrt().is_none(), "{a:?}");
        }
    }
}

/// gp lr(a, p) for the rows below: 59713600, 82062379, 116190042, 473844410, nonsq,
/// 2147483648, nonsq, 87776246428256997082236, nonsq, 1, 0, nonsq.
#[test]
fn prime_field_sqrt_matches_pari() {
    let rows: [(i64, &str, Option<&str>); 12] = [
        (2, "1000000007", Some("59713600")),
        (3, "1000000007", Some("82062379")),
        (7, "998244353", Some("116190042")),
        (17, "998244353", Some("473844410")),
        (5, "998244353", None),
        (2, "2305843009213693951", Some("2147483648")),
        (3, "2305843009213693951", None),
        (
            11,
            "1208925819614629174706189",
            Some("87776246428256997082236"),
        ),
        (2, "1208925819614629174706189", None),
        (1, "2", Some("1")),
        (0, "7", Some("0")),
        (6, "7", None),
    ];
    for (a, p, root) in rows {
        let x = PrimeField::new(int(a), big(p)).unwrap();
        assert_eq!(x.is_square(), root.is_some(), "{a} mod {p}");
        let got = x.sqrt();
        assert_eq!(
            got.as_ref().map(|r| r.value().clone()),
            root.map(big),
            "{a} mod {p}"
        );
        if let Some(r) = got {
            assert_eq!(r.modulus(), &big(p));
            assert!(r.clone() * r == x);
        }
    }
}

#[test]
fn prime_field_sqrt_sentinel_and_composite() {
    let one = <PrimeField as Ring>::one();
    let four = one.clone() + one.clone() + one.clone() + one.clone();
    let two = one.clone() + one.clone();
    let r = four.sqrt().expect("4 = 2^2 in Z");
    assert_eq!(r.modulus(), &Integer::zero());
    assert_eq!(r.value(), &int(2));
    assert!(two.sqrt().is_none());
    assert!((-one.clone()).sqrt().is_none());
    assert_eq!(
        <PrimeField as Ring>::zero().sqrt().unwrap().value(),
        &int(0)
    );
    // A zero element is a square for every modulus.
    assert_eq!(
        PrimeField::new(int(0), int(15))
            .unwrap()
            .sqrt()
            .unwrap()
            .value(),
        &int(0)
    );
}

#[test]
#[should_panic(expected = "PrimeField::sqrt: the modulus 15 is not prime")]
fn prime_field_sqrt_rejects_composite_modulus() {
    let _ = PrimeField::new(int(4), int(15)).unwrap().sqrt();
}

/// GF(9) = F_3[w]/(w^2 + 2w + 2) (Conway). gp: [[] sq []] [[1] sq [1]] [[2] sq [1, 1]]
/// [[0, 1] nonsq] [[1, 1] sq [0, 1]] [[2, 1] nonsq] [[0, 2] nonsq] [[1, 2] nonsq]
/// [[2, 2] sq [2, 1]].
#[test]
fn gf9_sqrt_matches_pari() {
    let f = FiniteField::new(int(3), 2).unwrap();
    assert_eq!(f.defining_polynomial(), &[int(2), int(2), int(1)][..]);
    check(&f, &["0", "0"], Some(&["0", "0"]));
    check(&f, &["1"], Some(&["1"]));
    check(&f, &["2"], Some(&["1", "1"]));
    check(&f, &["0", "1"], None);
    check(&f, &["1", "1"], Some(&["0", "1"]));
    check(&f, &["2", "1"], None);
    check(&f, &["0", "2"], None);
    check(&f, &["1", "2"], None);
    check(&f, &["2", "2"], Some(&["2", "1"]));
}

/// GF(125) = F_5[w]/(w^3 + 3w + 3) (Conway). gp: [[0, 1] nonsq] [[1, 1] sq [2, 3, 1]]
/// [[3, 0, 2] sq [2, 1, 1]] [[0, 0, 1] sq [0, 1]] [[3] nonsq] [[2] nonsq].
/// GF(256) = F_2[w]/(w^8 + w^4 + w^3 + w^2 + 1) (Conway). gp: sqrt(w) = [1, 0, 1, 0, 0, 0, 0, 1],
/// sqrt(w+1) = [0, 0, 1, 0, 0, 0, 0, 1].
#[test]
fn gf125_and_gf256_sqrt_match_pari() {
    let f = FiniteField::new(int(5), 3).unwrap();
    assert_eq!(
        f.defining_polynomial(),
        &[int(3), int(3), int(0), int(1)][..]
    );
    check(&f, &["0", "1"], None);
    check(&f, &["1", "1"], Some(&["2", "3", "1"]));
    check(&f, &["3", "0", "2"], Some(&["2", "1", "1"]));
    check(&f, &["0", "0", "1"], Some(&["0", "1"]));
    check(&f, &["3"], None);
    check(&f, &["2"], None);

    let g = FiniteField::new(int(2), 8).unwrap();
    let mut m = vec![int(0); 9];
    for i in [0usize, 2, 3, 4, 8] {
        m[i] = int(1);
    }
    assert_eq!(g.defining_polynomial(), &m[..]);
    check(
        &g,
        &["0", "1"],
        Some(&["1", "0", "1", "0", "0", "0", "0", "1"]),
    );
    check(
        &g,
        &["1", "1"],
        Some(&["0", "0", "1", "0", "0", "0", "0", "1"]),
    );
}

/// gp (show, see the header):
/// GF((2^61-1)^2), t^2 - 3: [[0, 1] sq [691116165103479690, 369169261380160233]] [[1, 1] nonsq]
///   [[2, 1] sq [1073741824, 1073741824]] [[5, 3] nonsq]
///   [[3, 7] sq [1014660761467108305, 511037510009013275]] [[3] sq [0, 1]]
///   [[9876543210, 12345678901234567] nonsq]
/// GF((2^80+13)^2), t^2 + t + 1: [[0, 1] sq [1, 1]] [[1, 1] sq [0, 453758120640033424617241]]
///   [[2, 1] nonsq] [[5, 3] nonsq] [[7] sq [690036740188608601964968, 171147660762588029223747]]
///   [[2] sq [794203387975751878202313, 379480956336874581698437]]
#[test]
fn large_field_sqrt_matches_pari() {
    let p = big("2305843009213693951");
    let f = FiniteField::with_modulus(p.clone(), vec![p.clone() - int(3), int(0), int(1)]).unwrap();
    check(
        &f,
        &["0", "1"],
        Some(&["691116165103479690", "369169261380160233"]),
    );
    check(&f, &["1", "1"], None);
    check(&f, &["2", "1"], Some(&["1073741824", "1073741824"]));
    check(&f, &["5", "3"], None);
    check(
        &f,
        &["3", "7"],
        Some(&["1014660761467108305", "511037510009013275"]),
    );
    check(&f, &["3"], Some(&["0", "1"]));
    check(&f, &["9876543210", "12345678901234567"], None);

    let p = big("1208925819614629174706189");
    let g = FiniteField::with_modulus(p, vec![int(1), int(1), int(1)]).unwrap();
    check(&g, &["0", "1"], Some(&["1", "1"]));
    check(&g, &["1", "1"], Some(&["0", "453758120640033424617241"]));
    check(&g, &["2", "1"], None);
    check(&g, &["5", "3"], None);
    check(
        &g,
        &["7"],
        Some(&["690036740188608601964968", "171147660762588029223747"]),
    );
    check(
        &g,
        &["2"],
        Some(&["794203387975751878202313", "379480956336874581698437"]),
    );
}

/// Every element of small fields: `is_square` is membership in {x^2}, every root squares back,
/// there are (q + 1)/2 squares for odd q and q for even q, and for n = 1 the root is the least
/// one in [0, p) (as `IntegerMod::sqrt`).
#[test]
fn finite_field_sqrt_exhaustive_small_fields() {
    let mut fields: Vec<(i64, usize)> = vec![(3, 1), (5, 1), (7, 1), (13, 1), (17, 1)];
    fields.extend((1..=4).map(|n| (2, n)));
    fields.extend((2..=5).map(|n| (3, n)));
    fields.extend([
        (5, 2),
        (5, 3),
        (7, 2),
        (11, 2),
        (13, 2),
        (17, 2),
        (41, 2),
        (97, 2),
    ]);
    for (p, n) in fields {
        let f = FiniteField::new(int(p), n).unwrap();
        let q = p.pow(n as u32);
        let elems: Vec<FiniteFieldElement> = (0..q)
            .map(|i| {
                let mut c = Vec::with_capacity(n);
                let mut t = i;
                for _ in 0..n {
                    c.push(int(t % p));
                    t /= p;
                }
                f.element(c)
            })
            .collect();
        let squares: HashSet<Vec<Integer>> = elems
            .iter()
            .map(|x| seq(&(x.clone() * x.clone())))
            .collect();
        let want = if p == 2 { q } else { (q + 1) / 2 };
        assert_eq!(squares.len() as i64, want, "GF({p}^{n})");
        for x in &elems {
            let is_sq = squares.contains(&seq(x));
            assert_eq!(x.is_square(), is_sq, "GF({p}^{n}) {:?}", x.eltseq());
            match x.sqrt() {
                Some(r) => {
                    assert!(is_sq);
                    assert!(r.clone() * r.clone() == *x);
                    if n == 1 {
                        let v = &seq(&r)[0];
                        let other = (int(p) - v.clone()) % int(p);
                        assert!(*v <= other, "least root in GF({p})");
                    }
                }
                None => assert!(!is_sq),
            }
        }
    }
}

#[test]
#[should_panic(expected = "needs a parent field")]
fn finite_field_sqrt_unbound_panics() {
    let _ = <FiniteFieldElement as Ring>::one().sqrt();
}

/// The legacy wrapper forwards: in GF(9) (Conway w^2 + 2w + 2), sqrt(2) = 1 + w (gp row
/// [[2] sq [1, 1]] above).
#[test]
fn extension_field_sqrt_forwards() {
    let irr = UnivariatePolynomial::new(vec![int(2), int(2), int(1)]);
    let two =
        ExtensionField::new(UnivariatePolynomial::new(vec![int(2)]), int(3), irr.clone()).unwrap();
    assert!(two.is_square());
    let r = two.sqrt().unwrap();
    assert_eq!(r.poly(), &UnivariatePolynomial::new(vec![int(1), int(1)]));
    let w =
        ExtensionField::new(UnivariatePolynomial::new(vec![int(0), int(1)]), int(3), irr).unwrap();
    assert!(!w.is_square());
    assert!(w.sqrt().is_none());
}
