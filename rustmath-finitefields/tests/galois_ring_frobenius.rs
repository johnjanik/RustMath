//! Tests of the Frobenius matrix, traces, orbit polynomials and Frobenius-ordered roots
//! (`galois_ring_frobenius`), and of the lazy product and the defining-polynomial normalization
//! of `GaloisRing` (MAGMA ch. 47 `GaloisImage`, `Trace`; ch. 48).
//!
//! Reference values: PARI/GP 2.15.4. `small_lift.gp` (design SL, `gp/`) orders the roots
//! computed by `polrootspadic` over `Q_p[t]/(f(t))` by `t^(p^i)` mod `p`, a method independent of
//! the Newton lift under test (`small_lift2.gp` does the same for a quintic). `ff_refs.gp`
//! (implementation report SL, `gp/`) gives the factorizations, the inert primes of
//! `x^23 - x - 1` and the large-prime image (Newton in `Mod(Mod(., p^N), f)`). The Teichmueller-digit Frobenius
//! `GaloisRingElement::frobenius` is a second, independent Frobenius inside RustMath.

use rustmath_core::Ring;
use rustmath_finitefields::{
    orbit_polynomial, FrobeniusMatrix, FrobeniusRoots, FrobeniusRootsError, GaloisRing,
    GaloisRingElement, NotInBaseRing,
};
use rustmath_integers::Integer;

fn int(v: i64) -> Integer {
    Integer::from(v)
}

fn ints(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&c| int(c)).collect()
}

fn big(s: &str) -> Integer {
    Integer::from_decimal_str(s).expect("decimal")
}

/// A small deterministic generator (splitmix64).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    /// A uniform-ish integer in [0, m) from enough random words.
    fn below(&mut self, m: &Integer) -> Integer {
        let words = (m.bit_length() / 64 + 2) as usize;
        let mut acc = Integer::zero();
        let base = Integer::from(1u64 << 32) * Integer::from(1u64 << 32);
        for _ in 0..words {
            acc = acc * base.clone() + Integer::from(self.next());
        }
        acc.modulo(m)
    }
    fn element(&mut self, r: &GaloisRing) -> GaloisRingElement {
        let m = r.characteristic().clone();
        r.element((0..r.degree()).map(|_| self.below(&m)).collect())
    }
}

/// The pre-change schoolbook product of `GaloisRingElement::mul`: a reduction mod `p^a` after
/// every addition, then the reduction by `D` from the top degree (the old `reduce`).
fn schoolbook(a: &[Integer], b: &[Integer], d_poly: &[Integer], m: &Integer) -> Vec<Integer> {
    let d = d_poly.len() - 1;
    let mut prod = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            prod[i + j] = (prod[i + j].clone() + ai.clone() * bj.clone()).modulo(m);
        }
    }
    for k in (d..prod.len()).rev() {
        let lead = prod[k].clone();
        if lead.is_zero() {
            continue;
        }
        for i in 0..d {
            prod[k - d + i] =
                (prod[k - d + i].clone() - lead.clone() * d_poly[i].clone()).modulo(m);
        }
        prod[k] = Integer::zero();
    }
    prod.truncate(d);
    prod
}

/// `x^23 - x - 1` (Galois group S23, H. Osada, J. Number Theory 25 (1987) 230-238).
fn trinomial23() -> Vec<Integer> {
    let mut g = vec![Integer::zero(); 24];
    g[0] = int(-1);
    g[1] = int(-1);
    g[23] = int(1);
    g
}

#[test]
fn lazy_mul_equals_schoolbook() {
    let mut rng = Rng(20261115);
    let mut rings = vec![
        GaloisRing::new(int(2), 8, 4).unwrap(),
        GaloisRing::new(int(5), 4, 3).unwrap(),
        GaloisRing::new(int(3), 20, 5).unwrap(),
        GaloisRing::new(int(29), 31, 1).unwrap(),
        GaloisRing::with_modulus(int(157), 20, trinomial23()).unwrap(),
    ];
    // A modulus with full-size coefficients: D = D0 + p * (random), D0 irreducible mod p.
    let base = GaloisRing::new(int(7), 1, 6).unwrap();
    let pa = int(7).pow(60);
    let mut d_big: Vec<Integer> = base.defining_polynomial().to_vec();
    for c in d_big.iter_mut().take(6) {
        *c = c.clone() + int(7) * rng.below(&pa);
    }
    rings.push(GaloisRing::with_modulus(int(7), 60, d_big).unwrap());
    for r in &rings {
        let m = r.characteristic().clone();
        let dp = r.defining_polynomial().to_vec();
        for _ in 0..12 {
            let a = rng.element(r);
            let b = rng.element(r);
            let want = schoolbook(a.eltseq(), b.eltseq(), &dp, &m);
            assert_eq!((a * b).eltseq(), want.as_slice(), "ring {r}");
        }
    }
}

#[test]
fn with_modulus_normalizes_a_leading_coefficient_one_mod_p() {
    // GR(5^2, 6x^2 + x + 2): 6 = 1 mod 5 but 6 != 1 mod 25. 6^-1 = 21 mod 25, so the stored
    // polynomial is 21 (6x^2 + x + 2) = x^2 + 21x + 17 mod 25 (review 1 of design SL, m5).
    let r = GaloisRing::with_modulus(int(5), 2, ints(&[2, 1, 6])).unwrap();
    assert_eq!(r.defining_polynomial(), ints(&[17, 21, 1]).as_slice());
    let w = r.generator();
    let dw = w.clone() * w.clone() + r.from_int(int(21)) * w.clone() + r.from_int(int(17));
    assert!(dw.is_zero(), "the generator is a root of the stored D");
    // The ring is (Z/25)[x]/(6x^2 + x + 2): 6w^2 + w + 2 = 0 as well.
    let raw = r.from_int(int(6)) * w.clone() * w.clone() + w + r.from_int(int(2));
    assert!(raw.is_zero());
    // A monic D is stored unchanged.
    let s = GaloisRing::with_modulus(int(5), 2, ints(&[2, 1, 1])).unwrap();
    assert_eq!(s.defining_polynomial(), ints(&[2, 1, 1]).as_slice());
}

#[test]
fn small_roots_match_pari() {
    // small_lift.gp: x^3 + x + 1 at p = 5, N = 6.
    let r = FrobeniusRoots::new(&ints(&[1, 1, 0, 1]), &int(5), 6).unwrap();
    let got: Vec<Vec<Integer>> = r.roots().iter().map(|e| e.eltseq().to_vec()).collect();
    assert_eq!(
        got,
        vec![
            ints(&[0, 1, 0]),
            ints(&[15371, 571, 15244]),
            ints(&[254, 15053, 381])
        ]
    );
    assert_eq!(r.modulus(), &int(15625));
    // small_lift.gp: x^23 - x - 1, first inert prime 23 (p equal to the degree), N = 4: phi(x).
    let r = FrobeniusRoots::new(&trinomial23(), &int(23), 4).unwrap();
    let want = ints(&[
        1, 49750, 206034, 90114, 100901, 269583, 26358, 48369, 44367, 15226, 208265, 6831, 272918,
        231541, 78177, 274482, 214475, 84019, 136275, 101959, 72128, 18768, 12719,
    ]);
    assert_eq!(r.roots()[1].eltseq(), want.as_slice());
    // small_lift2.gp (polrootspadic, as small_lift.gp): x^5 - x + 1 at 3, N = 7.
    let r = FrobeniusRoots::new(&ints(&[1, -1, 0, 0, 0, 1]), &int(3), 7).unwrap();
    assert_eq!(
        r.roots()[1].eltseq(),
        ints(&[807, 549, 1062, 1438, 1725]).as_slice()
    );
}

#[test]
fn roots_above_the_word_size_bound_match_pari() {
    // ff_refs.gp: 3037000573 is the first prime above fp_factor::MAX_P = 3037000499 at which
    // x^3 - x - 1 is irreducible; phi(t) mod p^2 by Newton in Mod(Mod(., p^2), t^3 - t - 1).
    let p = int(3037000573);
    let r = FrobeniusRoots::new(&ints(&[-1, -1, 0, 1]), &p, 2).unwrap();
    assert_eq!(
        r.roots()[1].eltseq(),
        &[
            big("2147444016801937653"),
            big("7137592157904941801"),
            big("1390520214998257685")
        ]
    );
    // x^2 + 1 at 3037000507 = 3 mod 4: phi(i) = i^p = -i.
    let p = int(3037000507);
    let r = FrobeniusRoots::new(&ints(&[1, 0, 1]), &p, 3).unwrap();
    assert_eq!(r.roots()[1], -r.roots()[0].clone());
}

#[test]
fn newton_image_equals_teichmuller_frobenius() {
    let mut rng = Rng(7);
    for r in [
        GaloisRing::new(int(5), 4, 3).unwrap(),
        GaloisRing::new(int(2), 8, 4).unwrap(),
    ] {
        let fm = FrobeniusMatrix::new(&r).unwrap();
        assert_eq!(fm.image_of_generator(), &r.generator().frobenius());
        for _ in 0..6 {
            let a = rng.element(&r);
            assert_eq!(fm.apply(&a), a.frobenius(), "ring {r}");
        }
    }
    let r = GaloisRing::with_modulus(int(23), 3, trinomial23()).unwrap();
    let fm = FrobeniusMatrix::new(&r).unwrap();
    assert_eq!(fm.image_of_generator(), &r.generator().frobenius());
    let a = rng.element(&r);
    assert_eq!(fm.apply(&a), a.frobenius());
}

#[test]
fn frobenius_is_a_ring_automorphism_of_order_d() {
    let mut rng = Rng(11);
    for r in [
        GaloisRing::new(int(3), 9, 4).unwrap(),
        GaloisRing::new(int(2), 16, 5).unwrap(),
        GaloisRing::with_modulus(int(157), 6, trinomial23()).unwrap(),
    ] {
        let fm = FrobeniusMatrix::new(&r).unwrap();
        let d = r.degree();
        for _ in 0..4 {
            let a = rng.element(&r);
            let b = rng.element(&r);
            assert_eq!(
                fm.apply(&(a.clone() * b.clone())),
                fm.apply(&a) * fm.apply(&b)
            );
            assert_eq!(
                fm.apply(&(a.clone() + b.clone())),
                fm.apply(&a) + fm.apply(&b)
            );
            let orbit = fm.orbit(&a);
            assert_eq!(orbit.len(), d);
            assert_eq!(fm.apply(&orbit[d - 1]), a, "phi^d = id");
            // phi fixes the base ring and maps to the p-th power mod p.
            let k = r.from_int(rng.below(r.characteristic()));
            assert_eq!(fm.apply(&k), k);
            assert_eq!(fm.apply(&a).residue(), a.residue().frobenius());
        }
        // phi has order exactly d on the generator.
        let orbit = fm.generator_orbit();
        for (i, e) in orbit.iter().enumerate().skip(1) {
            assert_ne!(e, &r.generator(), "phi^{i}(w) = w with i < d");
        }
        // Unbound constants are bound into the ring.
        assert_eq!(fm.apply(&GaloisRingElement::one()), r.one());
    }
    // d = 1: the identity.
    let z = GaloisRing::new(int(5), 3, 1).unwrap();
    let fm = FrobeniusMatrix::new(&z).unwrap();
    let a = z.from_int(int(77));
    assert_eq!(fm.apply(&a), a);
    assert_eq!(fm.trace(&a), int(77));
}

#[test]
fn trace_is_the_orbit_sum() {
    let mut rng = Rng(13);
    for r in [
        GaloisRing::new(int(2), 10, 3).unwrap(),
        GaloisRing::new(int(7), 5, 4).unwrap(),
        GaloisRing::with_modulus(int(157), 5, trinomial23()).unwrap(),
    ] {
        let fm = FrobeniusMatrix::new(&r).unwrap();
        for _ in 0..4 {
            let a = rng.element(&r);
            let sum = fm.orbit(&a).into_iter().fold(r.zero(), |acc, e| acc + e);
            assert_eq!(sum.base_ring_value(), Some(fm.trace(&a)), "ring {r}");
        }
        // Tr(1) = d.
        assert_eq!(
            fm.trace(&r.one()),
            Integer::from(r.degree() as u64).modulo(r.characteristic())
        );
    }
    // The traces of the powers of x for x^3 + x + 1 are its power sums 3, 0, -2 (PARI
    // polsym(x^3+x+1, 2) = [3, 0, -2]).
    let r = GaloisRing::with_modulus(int(5), 6, ints(&[1, 1, 0, 1])).unwrap();
    let fm = FrobeniusMatrix::new(&r).unwrap();
    assert_eq!(fm.generator_power_traces(), ints(&[3, 0, 15623]).as_slice());
}

#[test]
fn orbit_polynomial_vanishes_on_the_orbit() {
    let mut rng = Rng(17);
    let r = GaloisRing::new(int(3), 8, 5).unwrap();
    let fm = FrobeniusMatrix::new(&r).unwrap();
    for _ in 0..3 {
        let a = rng.element(&r);
        let poly = fm.orbit_polynomial(&a).unwrap();
        assert_eq!(poly.len(), 6);
        assert!(poly[5].is_one());
        for e in fm.orbit(&a) {
            let mut acc = r.zero();
            for c in poly.iter().rev() {
                acc = acc * e.clone() + r.from_int(c.clone());
            }
            assert!(acc.is_zero());
        }
        // The constant term is (-1)^d times the norm, the Z^(d-1) coefficient minus the trace.
        let tr = fm.trace(&a);
        assert_eq!(
            (poly[4].clone() + tr).modulo(r.characteristic()),
            Integer::zero()
        );
    }
    // The generator's orbit polynomial is the defining polynomial.
    assert_eq!(
        fm.orbit_polynomial(&r.generator()).unwrap(),
        r.defining_polynomial().to_vec()
    );
    assert_eq!(orbit_polynomial(&[]).unwrap(), vec![Integer::one()]);
}

#[test]
fn orbit_polynomial_rejects_an_unstable_family() {
    let r = GaloisRing::new(int(5), 3, 3).unwrap();
    let w = r.generator();
    // (Z - w)(Z - w - 1): the constant term w^2 + w is not in Z/125.
    let fam = [w.clone(), w.clone() + r.one()];
    assert_eq!(orbit_polynomial(&fam), Err(NotInBaseRing { index: 0 }));
    // (Z - w)(Z + w) = Z^2 - w^2: the constant term -w^2 is not in Z/125.
    let fam = [w.clone(), -w.clone()];
    assert_eq!(orbit_polynomial(&fam), Err(NotInBaseRing { index: 0 }));
    // A stable family passes.
    let fm = FrobeniusMatrix::new(&r).unwrap();
    assert!(orbit_polynomial(&fm.orbit(&(w.clone() + r.one()))).is_ok());
}

#[test]
fn with_precision_projects_and_continues() {
    // 23 and 157 are the inert primes of x^23 - x - 1 below 400 (ff_refs.gp).
    let f = trinomial23();
    let low = FrobeniusRoots::new(&f, &int(157), 3).unwrap();
    let high = low.with_precision(11).unwrap();
    let direct = FrobeniusRoots::new(&f, &int(157), 11).unwrap();
    assert_eq!(high.precision(), 11);
    for (a, b) in high.roots().iter().zip(direct.roots()) {
        assert_eq!(a, b);
    }
    let back = direct.with_precision(3).unwrap();
    for (a, b) in back.roots().iter().zip(low.roots()) {
        assert_eq!(a, b);
    }
    assert_eq!(
        back.frobenius().generator_power_traces(),
        low.frobenius().generator_power_traces()
    );
    assert!(matches!(
        low.with_precision(0),
        Err(FrobeniusRootsError::ZeroPrecision)
    ));
    // FrobeniusMatrix: with_precision only projects; lift_to needs a lift of the ring.
    let fm = low.frobenius();
    assert!(fm.with_precision(4).is_err());
    assert!(fm.with_precision(0).is_err());
    let lower = GaloisRing::with_modulus(int(157), 2, f.clone()).unwrap();
    assert!(fm.lift_to(&lower).is_err());
    let mut other = f.clone();
    other[0] = int(1); // x^23 - x + 1, also inert at 157 (review 1 of design SL, 1.4)
    let wrong = GaloisRing::with_modulus(int(157), 6, other).unwrap();
    assert!(fm.lift_to(&wrong).is_err());
    let right = GaloisRing::with_modulus(int(157), 6, f.clone()).unwrap();
    let lifted = fm.lift_to(&right).unwrap();
    assert_eq!(
        lifted.with_precision(3).unwrap().generator_orbit(),
        fm.generator_orbit()
    );
}

#[test]
fn refusals() {
    use FrobeniusRootsError as E;
    let p5 = int(5);
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 0, 2]), &p5, 3),
        Err(E::NotMonic)
    ));
    assert!(matches!(FrobeniusRoots::new(&[], &p5, 3), Err(E::NotMonic)));
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 1]), &p5, 3),
        Err(E::DegreeTooSmall(1))
    ));
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 1, 0, 1]), &p5, 0),
        Err(E::ZeroPrecision)
    ));
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 1, 0, 1]), &int(25), 3),
        Err(E::Prime(_))
    ));
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 1, 0, 1]), &int(1), 3),
        Err(E::Prime(_))
    ));
    // x^2 + 1 = (x + 1)^2 mod 2 (ff_refs.gp).
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, 0, 1]), &int(2), 3),
        Err(E::NotSquarefreeModP)
    ));
    // x^3 - x - 1 = (x + 3)(x^2 + 2x + 3) mod 5 (ff_refs.gp).
    assert_eq!(
        FrobeniusRoots::new(&ints(&[-1, -1, 0, 1]), &p5, 3).unwrap_err(),
        E::NotInertModP {
            factor_degrees: Some(vec![1, 2])
        }
    );
    // Above MAX_P: (x - 1)^2 (x + 1) is never squarefree; x^2 - 1 is never inert; x^3 - x - 1
    // splits as 1 + 2 at 3037000507 (ff_refs.gp).
    let big_p = int(3037000507);
    assert!(matches!(
        FrobeniusRoots::new(&ints(&[1, -1, -1, 1]), &big_p, 2),
        Err(E::NotSquarefreeModP)
    ));
    assert_eq!(
        FrobeniusRoots::new(&ints(&[-1, 0, 1]), &big_p, 2).unwrap_err(),
        E::NotInertModP {
            factor_degrees: None
        }
    );
    assert_eq!(
        FrobeniusRoots::new(&ints(&[-1, -1, 0, 1]), &big_p, 2).unwrap_err(),
        E::NotInertModP {
            factor_degrees: None
        }
    );
    // Display is informative.
    let e = FrobeniusRoots::new(&ints(&[-1, -1, 0, 1]), &p5, 3).unwrap_err();
    assert_eq!(e.to_string(), "f mod p is reducible, factor degrees [1, 2]");
}
