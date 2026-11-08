//! `rustmath-matrix` linear solving, vector spaces, F[x] Smith form, invariant factors,
//! minimal polynomial and rational form over prime fields (`PrimeField`).
//!
//! MAGMA Handbook ch. 26 §26.8 (`IsConsistent`, `Solution`), §26.11 (`MinimalPolynomial`),
//! §26.12.2 (`InvariantFactors`, `RationalForm`), §26.12.3 (`SmithForm`), and ch. 28 §28.3.2,
//! §28.6, §28.7 (`quo`, `ReduceVector`, `Coordinates`); digests
//! `rustmath-lava/docs/Chapter_26_Matrices.md` and `Chapter_28_Vector_Spaces.md`.
//!
//! Input entries are bound elements (`PrimeField::new(v, p)`), except in the `forms_sentinel_*`
//! tests, which build matrices on purpose from the unbound `Ring::zero()`/`Ring::one()`
//! sentinels (computed in Z semantics). Reference values: PARI/GP 2.15.4 and GAP 4, with the
//! command in a comment at each test.

use rustmath_finitefields::PrimeField;
use rustmath_integers::Integer;
use rustmath_matrix::companion::companion_matrix;
use rustmath_matrix::polynomial_matrix::PolynomialMatrix;
use rustmath_matrix::{
    charpoly_berkowitz, invariant_factors, minimal_polynomial, rational_canonical_form, Matrix,
    VectorSpace,
};
use rustmath_polynomials::UnivariatePolynomial;

fn gf(v: i64, p: i64) -> PrimeField {
    PrimeField::new(Integer::from(v), Integer::from(p)).unwrap()
}

fn gv(v: &[i64], p: i64) -> Vec<PrimeField> {
    v.iter().map(|&x| gf(x, p)).collect()
}

fn gm(rows: usize, cols: usize, v: &[i64], p: i64) -> Matrix<PrimeField> {
    Matrix::from_vec(rows, cols, gv(v, p)).unwrap()
}

/// The polynomial with coefficients `c` (low to high) over GF(p).
fn gp(c: &[i64], p: i64) -> UnivariatePolynomial<PrimeField> {
    UnivariatePolynomial::new(gv(c, p))
}

fn mat_vec(a: &Matrix<PrimeField>, x: &[PrimeField], p: i64) -> Vec<PrimeField> {
    (0..a.rows())
        .map(|i| {
            (0..a.cols()).fold(gf(0, p), |acc, j| {
                acc + a.get(i, j).unwrap().clone() * x[j].clone()
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// (a) Matrix::solve over GF(7)
// ---------------------------------------------------------------------------------------------

#[test]
fn solve_gf7() {
    // gp: G=Mod([1,2;2,4],7); matrank(G) = 1;
    //     matinverseimage(G,Mod([1,3]~,7)) = []~ (matrank of [G|b] = 2);
    //     matinverseimage(G,Mod([3,6]~,7)) = [Mod(3, 7), Mod(0, 7)]~
    let a = gm(2, 2, &[1, 2, 2, 4], 7);
    assert_eq!(a.solve(&gv(&[1, 3], 7)).unwrap(), None);
    let b = gv(&[3, 6], 7);
    let x = a.solve(&b).unwrap().expect("consistent");
    assert_eq!(x, gv(&[3, 0], 7));
    assert_eq!(mat_vec(&a, &x, 7), b);

    // gp: H=Mod([3,1,4;1,5,9;2,6,5],7); matdet(H) = Mod(1, 7);
    //     matsolve(H,Mod([1,2,3]~,7)) = [Mod(4, 7), Mod(0, 7), Mod(6, 7)]~
    let h = gm(3, 3, &[3, 1, 4, 1, 5, 9, 2, 6, 5], 7);
    let b = gv(&[1, 2, 3], 7);
    let x = h.solve(&b).unwrap().expect("consistent");
    assert_eq!(x, gv(&[4, 0, 6], 7));
    assert_eq!(mat_vec(&h, &x, 7), b);
}

// ---------------------------------------------------------------------------------------------
// (c) Smith form over GF(3)[x], invariant factors and rational form over GF(5), GF(7)
// ---------------------------------------------------------------------------------------------

#[test]
fn smith_form_gf3() {
    // gp: matsnf(Mod(1,3)*[x^2+1, x; x, x^2+2*x]) =
    //     [Mod(1, 3)*x^4 + Mod(2, 3)*x^3 + Mod(2, 3)*x, 1]
    let p = 3;
    let m = PolynomialMatrix::new(
        2,
        2,
        vec![
            gp(&[1, 0, 1], p),
            gp(&[0, 1], p),
            gp(&[0, 1], p),
            gp(&[0, 2, 1], p),
        ],
    )
    .unwrap();
    let snf = m.smith_normal_form().unwrap();
    let s = &snf.diagonal;
    assert_eq!(
        &snf.left_transform
            .mul(&m)
            .unwrap()
            .mul(&snf.right_transform)
            .unwrap(),
        s
    );
    assert_eq!(
        snf.left_transform.mul(&snf.left_inverse).unwrap(),
        PolynomialMatrix::identity(2)
    );
    assert!(s.get(0, 1).is_zero() && s.get(1, 0).is_zero());
    assert_eq!(s.get(0, 0), &gp(&[1], p));
    assert_eq!(s.get(1, 1), &gp(&[0, 2, 0, 2, 1], p));
}

/// Invariant factors, minimal polynomial, rational form with its certified change of basis,
/// and the product of the factors against `charpoly_berkowitz`.
fn check_forms(a: &Matrix<PrimeField>, expected: &[UnivariatePolynomial<PrimeField>], p: i64) {
    let n = a.rows();
    assert_eq!(invariant_factors(a).unwrap(), expected);
    assert_eq!(&minimal_polynomial(a).unwrap(), expected.last().unwrap());

    let rcf = rational_canonical_form(a).unwrap();
    assert_eq!(rcf.invariant_factors, expected);
    assert_eq!(rcf.minimal_polynomial(), expected.last());
    let mut c = gm(n, n, &vec![0; n * n], p);
    let mut off = 0;
    for f in expected {
        let b = companion_matrix(f).unwrap();
        for i in 0..b.rows() {
            for j in 0..b.cols() {
                c.set(off + i, off + j, b.get(i, j).unwrap().clone())
                    .unwrap();
            }
        }
        off += b.rows();
    }
    assert_eq!(off, n);
    assert_eq!(rcf.canonical_matrix, c);
    let t = rcf.change_of_basis.clone().expect("T is computed");
    assert_eq!(a.mul(&t).unwrap(), t.mul(&c).unwrap(), "A·T != T·C");
    assert!(t.inverse().unwrap().is_some(), "T singular");

    let product = expected.iter().fold(gp(&[1], p), |acc, f| acc * f.clone());
    assert_eq!(product, charpoly_berkowitz(a).unwrap());
}

#[test]
fn rational_form_gf5_derogatory() {
    // A = V5*(C(x^2+2) + C((x^2+2)(x+1)))*V5^-1 mod 5, V5 = [1,0,2,0,0;0,1,0,0,3;0,0,1,0,0;
    // 4,0,0,1,0;0,0,0,0,1] (det 1); x^2+2 is irreducible mod 5
    // (gp: factor(Mod(1,5)*(x^2+2)) = Mat([Mod(1, 5)*x^2 + Mod(2, 5), 1])).
    // gp: matfrobenius(A,1) = [x^3 + Mod(1, 5)*x^2 + Mod(2, 5)*x + Mod(2, 5), x^2 + Mod(2, 5)],
    //     minpoly(A) = x^3 + x^2 + 2*x + 2, charpoly(A) = x^5 + x^4 + 4*x^3 + 4*x^2 + 4*x + 4
    let p = 5;
    let a = gm(
        5,
        5,
        &[
            0, 3, 0, 0, 2, 4, 0, 2, 3, 2, 0, 0, 0, 0, 3, 0, 2, 1, 0, 2, 1, 0, 3, 1, 4,
        ],
        p,
    );
    check_forms(&a, &[gp(&[2, 0, 1], p), gp(&[2, 2, 1, 1], p)], p);
    assert_eq!(charpoly_berkowitz(&a).unwrap(), gp(&[4, 4, 4, 4, 1, 1], p));
}

#[test]
fn rational_form_gf7_scalar() {
    // gp: matfrobenius(Mod(2,7)*matid(3),1) = [x + Mod(5, 7), x + Mod(5, 7), x + Mod(5, 7)],
    //     minpoly = Mod(1, 7)*x + Mod(5, 7)
    let p = 7;
    let a = gm(3, 3, &[2, 0, 0, 0, 2, 0, 0, 0, 2], p);
    let f = gp(&[5, 1], p);
    check_forms(&a, &[f.clone(), f.clone(), f], p);
}

// ---------------------------------------------------------------------------------------------
// Krylov path over GF(p): larger matrices, sparse bound matrices, sentinel-only matrices
// ---------------------------------------------------------------------------------------------
//
// Integer test matrices, row-major, reduced mod p (the same matrices as the rustmath-matrix
// tests over Q). DENSE12: entries in [-5, 5]. UPTRI14: upper triangular, diagonal 1..14, strict
// upper part in [-5, 5]. DEROG12: U·(C(x^3+x+1) ⊕ C(x-2))^{⊕3}·U⁻¹, U unimodular. NILP15:
// U·(J1(0) ⊕ … ⊕ J5(0))·U⁻¹, U unimodular. In gp, with v the entries of an n×n one:
// A = matrix(n, n, i, j, v[(i-1)*n+j]).

#[rustfmt::skip]
const DENSE12: [i64; 144] = [
    0, 2, -2, -1, 1, 3, -5, 3, 5, 1, -4, 1,
    -4, -5, 0, -1, 5, 1, 2, -4, -1, 0, 1, -5,
    1, 4, 0, -2, 1, 1, 2, 5, -3, -5, 1, 2,
    5, -4, 2, 2, 2, -4, 3, 1, 5, 3, -4, -4,
    5, 5, 0, -5, -1, 2, -3, -5, -1, 3, 1, -4,
    -1, 1, 5, -2, -5, -5, 3, 3, 0, -4, -3, -1,
    2, 0, -3, 3, 2, 0, -4, 2, 1, -5, 4, -2,
    3, -2, -2, 5, 2, 0, 2, 0, -4, 4, -2, -4,
    -1, 2, 0, 2, 4, -1, -5, -2, 3, 5, 5, -2,
    -4, 5, -2, -2, 1, -5, 5, 1, -4, 1, -5, 4,
    -5, -2, 0, 0, -2, 0, 2, -2, -2, 1, -2, -5,
    5, 5, -4, 4, -1, -5, -4, 4, 5, -2, 0, 0,
];
#[rustfmt::skip]
const UPTRI14: [i64; 196] = [
    1, 0, -4, -2, 4, 0, 4, 0, -4, -4, 1, -5, -5, -2,
    0, 2, 3, -3, 4, 0, 1, 0, -1, -5, 2, -5, 0, -1,
    0, 0, 3, 3, 1, -1, 4, -1, -1, 0, 4, 2, -3, 5,
    0, 0, 0, 4, -2, -5, -1, -1, 0, -1, 3, -1, 3, 1,
    0, 0, 0, 0, 5, 2, -1, -1, -3, -2, 4, 5, 4, -5,
    0, 0, 0, 0, 0, 6, -4, -4, 0, -4, 4, 4, -4, 2,
    0, 0, 0, 0, 0, 0, 7, 3, -4, -4, 0, 3, 3, -2,
    0, 0, 0, 0, 0, 0, 0, 8, -4, 1, -2, 4, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 9, -1, 2, -4, 5, -2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 10, -4, 5, 5, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 2, -3, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 1, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 4,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14,
];
#[rustfmt::skip]
const DEROG12: [i64; 144] = [
    2, -4, 7, 0, -12, 0, 0, 0, -21, -10, -2, 0,
    -2, 4, -9, 2, 12, 2, -2, 0, 16, 8, 1, 2,
    0, 1, 0, 0, 2, 0, 0, 0, 5, 2, 0, 0,
    0, 0, -1, 2, -4, 0, 0, 0, -1, -2, -2, 0,
    2, 2, 8, 1, 0, 1, -1, 0, 6, 1, -3, 1,
    -1, 2, -4, 1, 1, 1, -1, -4, 14, 3, -8, 1,
    -3, 2, -14, 3, 7, 2, -1, -2, 10, 3, -2, 3,
    0, -2, 2, -1, -6, -1, 1, 2, -12, -5, 5, -1,
    1, 0, 4, 0, -2, 0, 0, 0, -1, -1, -1, 0,
    -3, -4, -12, -2, -2, -2, 2, 0, -12, -3, 5, -2,
    0, 0, -1, 0, 2, 0, 0, 0, 0, 1, 0, 0,
    -4, -4, -16, -2, 4, -2, 2, 0, -12, -2, 6, 0,
];
#[rustfmt::skip]
const NILP15: [i64; 225] = [
    -1, 2, 1, -2, 4, -2, -3, -5, 0, -1, 0, -4, -1, 1, 0,
    -1, 4, 2, -1, 1, -2, -1, -2, 1, 0, 0, -1, -2, 0, 2,
    0, 0, 0, -1, -2, 0, 0, 2, 1, 0, 0, 2, 0, -1, 1,
    -1, 2, 1, -2, 4, -2, -3, -5, 0, -1, 0, -4, -1, 1, 0,
    1, -2, 0, -3, -5, 3, 1, 5, 3, -1, 0, 5, 1, -4, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, -2, -1, 2, -3, 3, 3, 4, 0, 1, 0, 3, 1, -1, 0,
    1, 0, 0, -3, -5, 2, 1, 5, 3, -1, 0, 5, 0, -4, 3,
    -2, 2, 1, -1, 6, -4, -4, -7, -1, 0, 0, -6, -1, 3, 1,
    -1, 0, 0, 1, 1, -2, -1, -1, -1, 1, 0, -1, 0, 2, -1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    -3, 8, 4, -2, 3, -6, -3, -5, 2, 0, 0, -3, -4, 1, 4,
    0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[test]
fn forms_dense12_gf7() {
    // T-a9. gp: lift(matfrobenius(dense12*Mod(1,7),1)) = [x^12 + 4*x^11 + 6*x^10 + 4*x^8
    //     + 3*x^7 + x^6 + 5*x^5 + 5*x^4 + 6*x^3 + x + 6] = lift(minpoly(.)) = lift(charpoly(.))
    let p = 7;
    let a = gm(12, 12, &DENSE12, p);
    check_forms(&a, &[gp(&[6, 1, 0, 6, 5, 5, 1, 3, 4, 0, 6, 4, 1], p)], p);
}

#[test]
fn forms_uptri14_gf2() {
    // T-a10. gp: lift(matfrobenius(uptri14*Mod(1,2),1)) = [x^8 + x^4, x^4 + x^2, x^2 + x],
    //     lift(charpoly(uptri14*Mod(1,2))) = x^14 + x^13 + x^12 + x^11 + x^10 + x^9 + x^8 + x^7
    let p = 2;
    let a = gm(14, 14, &UPTRI14, p);
    check_forms(
        &a,
        &[
            gp(&[0, 1, 1], p),
            gp(&[0, 0, 1, 0, 1], p),
            gp(&[0, 0, 0, 0, 1, 0, 0, 0, 1], p),
        ],
        p,
    );
    assert_eq!(
        charpoly_berkowitz(&a).unwrap(),
        gp(&[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], p)
    );
}

#[test]
fn forms_uptri14_gf3() {
    // T-a11. gp: lift(matfrobenius(uptri14*Mod(1,3),1)) = [x^9 + x^8 + 2*x^3 + 2*x^2,
    //     x^3 + 2*x, x^2 + 2*x]
    let p = 3;
    let a = gm(14, 14, &UPTRI14, p);
    check_forms(
        &a,
        &[
            gp(&[0, 2, 1], p),
            gp(&[0, 2, 0, 1], p),
            gp(&[0, 0, 2, 2, 0, 0, 0, 0, 1, 1], p),
        ],
        p,
    );
}

#[test]
fn forms_derog12_gf2_gf3() {
    // T-a12. gp: lift(matfrobenius(derog12*Mod(1,2),1)) = [x^4 + x^2 + x] x 3,
    //     lift(matfrobenius(derog12*Mod(1,3),1)) = [x^4 + x^3 + x^2 + 2*x + 1] x 3
    let f2 = gp(&[0, 1, 1, 0, 1], 2);
    check_forms(&gm(12, 12, &DEROG12, 2), &[f2.clone(), f2.clone(), f2], 2);
    let f3 = gp(&[1, 2, 1, 1, 1], 3);
    check_forms(&gm(12, 12, &DEROG12, 3), &[f3.clone(), f3.clone(), f3], 3);
}

#[test]
fn forms_nilp15_gf2() {
    // T-a13. gp: lift(matfrobenius(nilp15*Mod(1,2),1)) = [x^5, x^4, x^3, x^2, x]
    let p = 2;
    let expected: Vec<_> = (1..=5)
        .map(|k| {
            let mut c = vec![0; k + 1];
            c[k] = 1;
            gp(&c, p)
        })
        .collect();
    check_forms(&gm(15, 15, &NILP15, p), &expected, p);
}

#[test]
fn forms_sentinel_identity_and_zero() {
    // T-a14. Matrices built only from the unbound Ring::one()/zero() sentinels stay in Z
    // semantics (the field constants z, one, ±1 of the Krylov path stay sentinels too); their
    // factors compare equal to the bound ones for any p, since sentinel == coerces.
    // gp: matfrobenius(matid(3),1) = [x - 1, x - 1, x - 1], matfrobenius(matrix(3,3),1) =
    //     [x, x, x]; lift(matfrobenius(Mod(matid(3),2),1)) = [x + 1, x + 1, x + 1],
    //     lift(matfrobenius(Mod(matrix(3,3),5),1)) = [x, x, x]
    for p in [2, 5] {
        let id = Matrix::<PrimeField>::identity(3);
        let f = gp(&[-1, 1], p);
        check_forms(&id, &[f.clone(), f.clone(), f], p);
        let z = Matrix::<PrimeField>::zeros(3, 3);
        let x = gp(&[0, 1], p);
        check_forms(&z, &[x.clone(), x.clone(), x], p);
    }
}

#[test]
fn forms_sentinel_non_unit_pivot_is_err() {
    // A sentinel-only matrix is computed in Z, where only ±1 are invertible: [0,0; 2,0] built
    // from Ring::one() + Ring::one() needs the pivot 2 in the Krylov path and in the fallback
    // (the Smith form of xI - A), so every function returns Err (never a value, never a
    // panic). The answer depends on the field the entries are meant to lie in:
    // gp: lift(matfrobenius(Mod([0,0;2,0],2),1)) = [x, x],
    //     lift(matfrobenius(Mod([0,0;2,0],3),1)) = [x^2].
    use rustmath_core::Ring;
    let two = PrimeField::one() + PrimeField::one();
    let a = Matrix::from_vec(
        2,
        2,
        vec![
            PrimeField::zero(),
            PrimeField::zero(),
            two,
            PrimeField::zero(),
        ],
    )
    .unwrap();
    assert!(invariant_factors(&a).is_err());
    assert!(minimal_polynomial(&a).is_err());
    assert!(rational_canonical_form(&a).is_err());
    // Bound, the same matrix is fine.
    check_forms(
        &gm(2, 2, &[0, 0, 2, 0], 2),
        &[gp(&[0, 1], 2), gp(&[0, 1], 2)],
        2,
    );
    check_forms(&gm(2, 2, &[0, 0, 2, 0], 3), &[gp(&[0, 0, 1], 3)], 3);
}

/// The unbound (modulus-0) element `x` of Z: `±(Ring::one() + … + Ring::one())`.
fn sent(x: i64) -> PrimeField {
    use rustmath_core::Ring;
    let mut s = PrimeField::zero();
    for _ in 0..x.unsigned_abs() {
        s = if x > 0 {
            s + PrimeField::one()
        } else {
            s - PrimeField::one()
        };
    }
    s
}

/// The n×n matrix of sentinel entries `v` (row-major).
fn sm(n: usize, v: &[i64]) -> Matrix<PrimeField> {
    Matrix::from_vec(n, n, v.iter().map(|&x| sent(x)).collect()).unwrap()
}

/// The polynomial with sentinel coefficients `c` (low to high).
fn sp(c: &[i64]) -> UnivariatePolynomial<PrimeField> {
    UnivariatePolynomial::new(c.iter().map(|&x| sent(x)).collect())
}

/// For a sentinel-only `A` (computed in Z semantics): invariant factors, minimal polynomial
/// and rational form equal `expected` in Z, `C` is the block companion matrix, `A·T == T·C` in
/// Z, `det T = ±1` (the constant term of the division-free `charpoly_berkowitz(T)`, so `T` is
/// invertible over Z and mod every p), `T` stays unbound (Z semantics), and the product of the
/// factors is `charpoly_berkowitz(A)`. Then for p in {2, 3, 5, 7} the same matrix with bound
/// entries has factors that compare equal (sentinel coercion) to `expected`.
fn check_forms_sentinel(n: usize, v: &[i64], expected: &[UnivariatePolynomial<PrimeField>]) {
    let a = sm(n, v);
    assert_eq!(invariant_factors(&a).unwrap(), expected, "A = {v:?}");
    assert_eq!(
        &minimal_polynomial(&a).unwrap(),
        expected.last().unwrap(),
        "A = {v:?}"
    );
    let rcf = rational_canonical_form(&a).unwrap_or_else(|e| panic!("A = {v:?}: {e:?}"));
    assert_eq!(rcf.invariant_factors, expected, "A = {v:?}");
    let mut c = sm(n, &vec![0; n * n]);
    let mut off = 0;
    for f in expected {
        let b = companion_matrix(f).unwrap();
        for i in 0..b.rows() {
            for j in 0..b.cols() {
                c.set(off + i, off + j, b.get(i, j).unwrap().clone())
                    .unwrap();
            }
        }
        off += b.rows();
    }
    assert_eq!(off, n);
    assert_eq!(rcf.canonical_matrix, c, "A = {v:?}");
    let t = rcf.change_of_basis.clone().expect("T is computed");
    assert!(
        t.data().iter().all(|x| x.modulus().is_zero()),
        "A = {v:?}: T is not in Z semantics"
    );
    assert_eq!(
        a.mul(&t).unwrap(),
        t.mul(&c).unwrap(),
        "A = {v:?}: A·T != T·C"
    );
    let det_t = charpoly_berkowitz(&t).unwrap().coeff(0).clone();
    assert!(
        det_t == sent(1) || det_t == sent(-1),
        "A = {v:?}: det T = {det_t:?} is not ±1"
    );
    let product = expected.iter().fold(sp(&[1]), |acc, f| acc * f.clone());
    assert_eq!(product, charpoly_berkowitz(&a).unwrap(), "A = {v:?}");
    for p in [2, 3, 5, 7] {
        assert_eq!(
            invariant_factors(&gm(n, n, v, p)).unwrap(),
            expected,
            "p = {p}, A = {v:?}"
        );
    }
}

#[test]
fn forms_sentinel_permutations_and_jordan() {
    // Matrices of Ring::zero()/one() sentinels on which the Krylov path's ±1 start vectors
    // create a non-unit pivot in Z (e.g. 3-cycle: "No inverse exists: gcd(-2, 0) = 2"); all
    // three functions fall back to the Smith form of xI - A, whose pivots are units here.
    // gp: S3=[0,1,0;0,0,1;1,0,0]; matfrobenius(S3,1) = [x^3 - 1]
    check_forms_sentinel(3, &[0, 1, 0, 0, 0, 1, 1, 0, 0], &[sp(&[-1, 0, 0, 1])]);
    // T3=[0,1,0;1,0,0;0,0,1]; matfrobenius(T3,1) = [x^2 - 1, x - 1]
    check_forms_sentinel(
        3,
        &[0, 1, 0, 1, 0, 0, 0, 0, 1],
        &[sp(&[-1, 1]), sp(&[-1, 0, 1])],
    );
    // J3=[1,1,0;0,1,1;0,0,1]; matfrobenius(J3,1) = [x^3 - 3*x^2 + 3*x - 1]
    check_forms_sentinel(3, &[1, 1, 0, 0, 1, 1, 0, 0, 1], &[sp(&[-1, 3, -3, 1])]);
    // O3=matrix(3,3,i,j,1); matfrobenius(O3,1) = [x^2 - 3*x, x];
    //     lift(matfrobenius(O3*Mod(1,3),1)) = [x^2, x], lift(matfrobenius(O3*Mod(1,2),1)) =
    //     [x^2 + x, x] (both compare equal to the Z result, checked in check_forms_sentinel)
    check_forms_sentinel(3, &[1; 9], &[sp(&[0, 1]), sp(&[0, -3, 1])]);
    assert_eq!(
        invariant_factors(&sm(3, &[1; 9])).unwrap(),
        vec![gp(&[0, 1], 3), gp(&[0, 0, 1], 3)]
    );
    assert_eq!(
        invariant_factors(&sm(3, &[1; 9])).unwrap(),
        vec![gp(&[0, 1], 2), gp(&[0, 1, 1], 2)]
    );
    // N4=matrix(4,4,i,j,j==i+1); matfrobenius(N4,1) = [x^4]
    let mut v = vec![0; 16];
    for i in 0..3 {
        v[i * 4 + i + 1] = 1;
    }
    check_forms_sentinel(4, &v, &[sp(&[0, 0, 0, 0, 1])]);

    // The same fallback for the GF(p^n) element type's sentinels: the 3-cycle and the
    // all-ones 3×3 matrix (gp values as above).
    use rustmath_core::Ring;
    use rustmath_finitefields::FiniteFieldElement as E;
    let (o, z) = (E::one(), E::zero());
    let ep = |c: &[E]| UnivariatePolynomial::new(c.to_vec());
    let three = o.clone() + o.clone() + o.clone();
    let cases = [
        (
            vec![0, 1, 0, 0, 0, 1, 1, 0, 0],
            vec![ep(&[
                z.clone() - o.clone(),
                z.clone(),
                z.clone(),
                o.clone(),
            ])],
        ),
        (
            vec![1; 9],
            vec![
                ep(&[z.clone(), o.clone()]),
                ep(&[z.clone(), z.clone() - three, o.clone()]),
            ],
        ),
    ];
    for (v, expected) in cases {
        let a = Matrix::from_vec(
            3,
            3,
            v.iter()
                .map(|&x| if x == 1 { o.clone() } else { z.clone() })
                .collect(),
        )
        .unwrap();
        assert_eq!(invariant_factors(&a).unwrap(), expected, "A = {v:?}");
        assert_eq!(&minimal_polynomial(&a).unwrap(), expected.last().unwrap());
        let rcf = rational_canonical_form(&a).unwrap();
        assert_eq!(rcf.invariant_factors, expected, "A = {v:?}");
        let t = rcf.change_of_basis.expect("T is computed");
        assert_eq!(
            a.mul(&t).unwrap(),
            t.mul(&rcf.canonical_matrix).unwrap(),
            "A = {v:?}"
        );
    }
}

/// The natural sentinel-only test families of size n, with their names: cyclic shift,
/// reversal, transposition (1 2), all-ones, upper and lower all-ones, nilpotent shift,
/// `J_n(1)` and lower `J_n(-1)`.
fn sentinel_families(n: usize) -> Vec<(String, Vec<i64>)> {
    let mut fams = Vec::new();
    let mut v = vec![0; n * n];
    for i in 0..n {
        v[i * n + (i + 1) % n] = 1;
    }
    fams.push((format!("shift{n}"), v));
    let mut v = vec![0; n * n];
    for i in 0..n {
        v[i * n + n - 1 - i] = 1;
    }
    fams.push((format!("reversal{n}"), v));
    let mut v = vec![0; n * n];
    for i in 2..n {
        v[i * n + i] = 1;
    }
    v[1] = 1;
    v[n] = 1;
    fams.push((format!("transp{n}"), v));
    fams.push((format!("ones{n}"), vec![1; n * n]));
    let upper = (0..n * n).map(|k| i64::from(k / n <= k % n)).collect();
    fams.push((format!("upperones{n}"), upper));
    let lower = (0..n * n).map(|k| i64::from(k / n >= k % n)).collect();
    fams.push((format!("lowerones{n}"), lower));
    let mut v = vec![0; n * n];
    for i in 0..n - 1 {
        v[i * n + i + 1] = 1;
    }
    fams.push((format!("nilshift{n}"), v));
    let mut v = vec![0; n * n];
    for i in 0..n {
        v[i * n + i] = 1;
        if i + 1 < n {
            v[i * n + i + 1] = 1;
        }
    }
    fams.push((format!("J{n}(1)"), v));
    let mut v = vec![0; n * n];
    for i in 0..n {
        v[i * n + i] = -1;
        if i + 1 < n {
            v[(i + 1) * n + i] = 1;
        }
    }
    fams.push((format!("lowerJ{n}(-1)"), v));
    fams
}

#[test]
fn forms_sentinel_natural_families() {
    // The 45 natural sentinel-only matrices n = 2..6 of sentinel_families. Before the Krylov
    // path, the Smith form of xI - A answered 39 of them; the Krylov path alone answered only
    // 19 (review B1). With the fallback, 43 answer: those 39 and upperones4, lowerones4,
    // lowerones5, lowerones6 (Krylov path only). upperones5 and upperones6 need a non-unit
    // pivot in both paths and stay Err (they did before as well); for them only "Err or a
    // certified value" is asserted, so a later improvement does not break this test.
    // Every value is proved by check_forms_sentinel's certificate in Z (C a block companion
    // matrix of a divisibility chain, A·T == T·C, det T = ±1: A is similar to C over Z, so
    // over Q and mod every p) and cross-checked with the bound GF(2), GF(3), GF(5), GF(7)
    // results. gp, for each of the 43 with L the factors listed decreasingly:
    //     matfrobenius(A,1) == L, and matfrobenius(A*Mod(1,p),1) == Mod(1,p)*L for
    //     p = 2, 3, 5, 7 (172 comparisons, all equal).
    let mut answered = 0;
    for n in 2..=6usize {
        for (name, v) in sentinel_families(n) {
            let a = sm(n, &v);
            match invariant_factors(&a) {
                Ok(expected) => {
                    check_forms_sentinel(n, &v, &expected);
                    answered += 1;
                }
                Err(e) => {
                    assert!(
                        name == "upperones5" || name == "upperones6",
                        "{name}: {e:?}"
                    );
                    assert!(minimal_polynomial(&a).is_err(), "{name}");
                    assert!(rational_canonical_form(&a).is_err(), "{name}");
                }
            }
        }
    }
    assert!(answered >= 43, "only {answered} of 45 answered");
}

#[test]
fn forms_sparse_bound_gf2_gf3_gf5() {
    // Sparse bound matrices on which ±1 start vectors built from the bare sentinels would give
    // a hidden zero (1 + 1 = 2 in Z) and a spurious Err, or a panic in the certificate.
    // GF(2), 4×4, a single 1 at (2,3) (1-indexed).
    // gp: A2=matrix(4,4); A2[2,3]=1; lift(matfrobenius(A2*Mod(1,2),1)) = [x^2, x, x]
    let mut v = vec![0; 16];
    v[4 + 2] = 1;
    check_forms(
        &gm(4, 4, &v, 2),
        &[gp(&[0, 1], 2), gp(&[0, 1], 2), gp(&[0, 0, 1], 2)],
        2,
    );
    // GF(3), 5×5, A[2,1] = 2, A[2,2] = 1 (1-indexed).
    // gp: A3=matrix(5,5); A3[2,1]=2; A3[2,2]=1; lift(matfrobenius(A3*Mod(1,3),1)) =
    //     [x^2 + 2*x, x, x, x]
    let mut v = vec![0; 25];
    v[5] = 2;
    v[6] = 1;
    let x3 = gp(&[0, 1], 3);
    check_forms(
        &gm(5, 5, &v, 3),
        &[x3.clone(), x3.clone(), x3, gp(&[0, 2, 1], 3)],
        3,
    );
    // GF(5), 7×7. gp: A5=[0,0,0,0,0,0,0; 0,3,0,0,0,0,0; 4,2,0,0,0,0,0; 0,0,0,0,0,0,0;
    //     0,0,0,0,0,0,0; 0,0,2,0,0,0,0; 0,4,0,0,0,0,0];
    //     lift(matfrobenius(A5*Mod(1,5),1)) = [x^4 + 2*x^3, x, x, x]
    #[rustfmt::skip]
    let v = [
        0, 0, 0, 0, 0, 0, 0,
        0, 3, 0, 0, 0, 0, 0,
        4, 2, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 0, 0, 0,
        0, 4, 0, 0, 0, 0, 0,
    ];
    let x5 = gp(&[0, 1], 5);
    check_forms(
        &gm(7, 7, &v, 5),
        &[x5.clone(), x5.clone(), x5, gp(&[0, 0, 0, 2, 1], 5)],
        5,
    );
    // GF(5), 3×3: zero products in A·v are skipped, and every entry of T must still be bound
    // (each accumulator starts from the bound zero of A's field).
    // gp: lift(matfrobenius(Mod([0,1,0;0,0,0;2,0,3],5),1)) = [x^3 + 2*x^2]
    let a = gm(3, 3, &[0, 1, 0, 0, 0, 0, 2, 0, 3], 5);
    check_forms(&a, &[gp(&[0, 0, 2, 1], 5)], 5);
    let t = rational_canonical_form(&a)
        .unwrap()
        .change_of_basis
        .unwrap();
    assert!(t.data().iter().all(|x| x.modulus() == &Integer::from(5)));
}

/// The non-constant diagonal of the Smith form of `xI - A` over GF(p)[x] (an independent
/// oracle for the invariant factors).
fn smith_oracle(a: &Matrix<PrimeField>, p: i64) -> Vec<UnivariatePolynomial<PrimeField>> {
    let n = a.rows();
    let mut data = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let neg = gf(0, p) - a.get(i, j).unwrap().clone();
            data.push(if i == j {
                UnivariatePolynomial::new(vec![neg, gf(1, p)])
            } else {
                UnivariatePolynomial::new(vec![neg])
            });
        }
    }
    let snf = PolynomialMatrix::new(n, n, data)
        .unwrap()
        .smith_normal_form()
        .unwrap();
    (0..n)
        .map(|i| snf.diagonal.get(i, i).clone())
        .filter(|d| d.degree().is_some_and(|k| k > 0))
        .collect()
}

#[test]
fn forms_sparse_gfp_fuzz() {
    // 300 deterministic sparse bound matrices, n = 1..8, p in {2, 3, 5}, density 10-90 %: the
    // input class in which sentinel start vectors failed. For each: rational_canonical_form is
    // Ok (no panic), A·T == T·C with T invertible, invariant_factors equals the rational form's
    // factors and the Smith form of xI - A, the last factor is minimal_polynomial, and the
    // product of the factors is charpoly_berkowitz.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = |m: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % m
    };
    for t in 0..300usize {
        let p = [2i64, 3, 5][t % 3];
        let n = 1 + (t / 3) % 8;
        let density = 1 + next(9); // tenths
        let v: Vec<i64> = (0..n * n)
            .map(|_| {
                if next(10) < density {
                    1 + next(p as u64 - 1) as i64
                } else {
                    0
                }
            })
            .collect();
        let a = gm(n, n, &v, p);
        let rcf =
            rational_canonical_form(&a).unwrap_or_else(|e| panic!("p = {p}, A = {v:?}: {e:?}"));
        let t_ = rcf.change_of_basis.clone().expect("T is computed");
        assert_eq!(
            a.mul(&t_).unwrap(),
            t_.mul(&rcf.canonical_matrix).unwrap(),
            "p = {p}, A = {v:?}"
        );
        assert!(t_.inverse().unwrap().is_some(), "p = {p}, A = {v:?}");
        // Bound input gives a T bound to GF(p) throughout (no modulus-0 sentinel zeros).
        assert!(
            t_.data().iter().all(|x| x.modulus() == &Integer::from(p)),
            "p = {p}, A = {v:?}: T has unbound entries"
        );
        let fs = invariant_factors(&a).unwrap();
        assert_eq!(fs, rcf.invariant_factors, "p = {p}, A = {v:?}");
        assert_eq!(fs, smith_oracle(&a, p), "p = {p}, A = {v:?}");
        assert_eq!(fs.last(), Some(&minimal_polynomial(&a).unwrap()));
        let product = fs.iter().fold(gp(&[1], p), |acc, f| acc * f.clone());
        assert_eq!(
            product,
            charpoly_berkowitz(&a).unwrap(),
            "p = {p}, A = {v:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// (b) Handbook H28E10 / H28E13 shape: the ternary Golay code G3 in GF(3)^11
// ---------------------------------------------------------------------------------------------

/// The rows x^i·g, i = 0..5, with g = x^5 + x^4 + 2x^3 + x^2 + 2, which divides x^11 - 1 over
/// GF(3) (gp: factor(Mod(1,3)*(x^11-1)) contains x^5 + x^4 + 2*x^3 + x^2 + 2).
fn golay_basis() -> Vec<Vec<PrimeField>> {
    let rows: [[i64; 11]; 6] = [
        [2, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 2, 1, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0],
        [0, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 2, 1, 1],
    ];
    rows.iter().map(|r| gv(r, 3)).collect()
}

fn unit(i: usize) -> Vec<PrimeField> {
    let mut v = gv(&[0; 11], 3);
    v[i] = gf(1, 3);
    v
}

#[test]
fn golay_code_quotient_gf3() {
    // gp: matrank(G) = 6, 11 - matrank(G) = 5;
    //     matinverseimage(G~, w~) = [2, 0, 1, 0, 0, 1]~ for w = [2,0,1,0,0,1]*G;
    //     matsolve(matconcat([G~, E[,1..5]]), E[,11])[7..11] = [0, 1, 2, 1, 1]~ (det 1),
    //     ... E[,1])[7..11] = [1, 0, 0, 0, 0]~
    // GAP: C := Subspace(GF(3)^11, G); Dimension(C) = 6, Dimension(V) - Dimension(C) = 5;
    //     Coefficients(Basis(C, G), w) = [ 2, 0, 1, 0, 0, 1 ];
    //     SiftedVector(SemiEchelonBasis(C), e1) = [ 0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2 ],
    //     SiftedVector(SemiEchelonBasis(C), w) = 0
    let c = VectorSpace::from_basis(golay_basis()).unwrap();
    assert_eq!(c.dimension(), 6);
    assert_eq!(c.ambient_dimension(), 11);

    let w = gv(&[1, 0, 1, 1, 0, 0, 1, 2, 2, 1, 1], 3);
    assert_eq!(c.coordinates(&w).unwrap(), gv(&[2, 0, 1, 0, 0, 1], 3));
    assert!(c.contains(&w));
    assert!(!c.contains(&unit(0)));
    assert!(c.coordinates(&unit(0)).is_err());

    assert_eq!(
        c.reduce_vector(&unit(0)).unwrap(),
        gv(&[0, 0, 0, 0, 0, 0, 1, 0, 2, 1, 2], 3)
    );
    assert_eq!(c.reduce_vector(&w).unwrap(), gv(&[0; 11], 3));

    let v = VectorSpace::<PrimeField>::new(11);
    let q = v.quotient(&c).unwrap();
    assert_eq!(q.dimension(), 5);
    let expected_complement: Vec<Vec<PrimeField>> = (0..5).map(unit).collect();
    assert_eq!(q.complement_basis(), &expected_complement[..]);
    assert_eq!(q.project(&unit(0)).unwrap(), gv(&[1, 0, 0, 0, 0], 3));
    assert_eq!(q.project(&unit(10)).unwrap(), gv(&[0, 1, 2, 1, 1], 3));
    assert_eq!(q.project(&w).unwrap(), gv(&[0, 0, 0, 0, 0], 3));
}
