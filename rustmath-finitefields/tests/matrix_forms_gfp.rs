//! `rustmath-matrix` linear solving, vector spaces, F[x] Smith form, invariant factors,
//! minimal polynomial and rational form over prime fields (`PrimeField`).
//!
//! MAGMA Handbook ch. 26 §26.8 (`IsConsistent`, `Solution`), §26.11 (`MinimalPolynomial`),
//! §26.12.2 (`InvariantFactors`, `RationalForm`), §26.12.3 (`SmithForm`), and ch. 28 §28.3.2,
//! §28.6, §28.7 (`quo`, `ReduceVector`, `Coordinates`); digests
//! `rustmath-lava/docs/Chapter_26_Matrices.md` and `Chapter_28_Vector_Spaces.md`.
//!
//! Every input entry is a bound element (`PrimeField::new(v, p)`). Reference values: PARI/GP
//! 2.15.4 and GAP 4, with the command in a comment at each test.

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
