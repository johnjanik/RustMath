//! P0 Mestre toolchain validation: the bivariate pencil discriminant vs PARI, and
//! the real A_24 seed g_20 = L·u (deg 24, r=20, square discriminant).

use rustmath_integers::Integer;
use rustmath_polynomials::bivariate::poly_sqrt;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::mestre::pencil_discriminant;
use rustmath_polynomials::real_roots::count_real_roots_int;
use rustmath_rationals::Rational;

fn qs(v: &[i64]) -> Vec<Rational> {
    v.iter().map(|&x| Rational::from(x)).collect()
}
fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}
fn imul(a: &[Integer], b: &[Integer]) -> Vec<Integer> {
    let mut o = vec![Integer::zero(); a.len() + b.len() - 1];
    for (i, ca) in a.iter().enumerate() {
        for (j, cb) in b.iter().enumerate() {
            o[i + j] = o[i + j].clone() + ca.clone() * cb.clone();
        }
    }
    o
}

#[test]
fn pencil_discriminant_matches_pari_degree5() {
    // P = x^5 - x, H = 2x^4 + x^2 + 3; PARI poldisc(P - T*H, x) =
    // 203136 T^8 + 195196 T^6 + 372624 T^4 + 4416 T^2 - 256
    let p = qs(&[0, -1, 0, 0, 0, 1]);
    let h = qs(&[3, 0, 1, 0, 2]);
    let d = pencil_discriminant(&p, &h);
    let expect = qs(&[-256, 0, 4416, 0, 372624, 0, 195196, 0, 203136]);
    assert_eq!(d, expect);
}

#[test]
fn mestre_seed_g20_is_valid() {
    // u(X) = X^4 - 5X^3 + 2X^2 + 9X + 10  (disc 8281 = 91^2, no real roots)
    let u = iz(&[10, 9, 2, -5, 1]);
    // L(X) = prod_{i=1}^{10} (X^2 - i^2)
    let mut l = iz(&[1]);
    for i in 1..=10i64 {
        l = imul(&l, &iz(&[-i * i, 0, 1]));
    }
    let g20 = imul(&l, &u);
    // deg 24
    assert_eq!(g20.len() - 1, 24);
    // exactly 20 real roots (the 20 rational roots ±1..±10; u has none)
    assert_eq!(count_real_roots_int(&g20), 20);
    // square discriminant  →  Gal ⊆ A_24
    let dg = discriminant(&g20);
    assert!(dg.is_perfect_square(), "disc(g20) must be a perfect square");

    // Step 2: P = X·g20 is separable degree 25 with square discriminant
    let mut p = vec![Integer::zero()];
    p.extend(g20.iter().cloned()); // X · g20
    assert_eq!(p.len() - 1, 25);
    let dp = discriminant(&p);
    assert!(dp.is_perfect_square(), "disc(X·g20) must be a perfect square");
}
