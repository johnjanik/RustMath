//! THROWAWAY adversarial-verification tests — DELETE THIS FILE AFTER RUNNING.
//! Expected values derived independently with sympy before writing.

use rustmath_integers::Integer;
use rustmath_matrix::lattice::{lll_is_reduced_exact, lll_reduce_real};
use rustmath_matrix::{charpoly_berkowitz, Matrix};
use rustmath_rationals::Rational;
use rustmath_reals::bigfloat::BigFloat;

fn z(n: i64) -> Integer {
    Integer::from(n)
}
fn q(n: i64, d: i64) -> Rational {
    Rational::new(n, d).unwrap()
}

#[test]
fn adv_charpoly_4x4_integer_vs_sympy() {
    // sympy: charpoly of [[3,-1,2,0],[5,4,-2,1],[0,2,1,-3],[-2,1,0,6]]
    //   = x^4 - 14x^3 + 75x^2 - 233x + 385
    let data: Vec<Integer> = [3, -1, 2, 0, 5, 4, -2, 1, 0, 2, 1, -3, -2, 1, 0, 6]
        .iter()
        .map(|&v| z(v))
        .collect();
    let a = Matrix::from_vec(4, 4, data).unwrap();
    let p = charpoly_berkowitz(&a).unwrap();
    let c = p.coefficients();
    let expect: [i64; 5] = [385, -233, 75, -14, 1]; // low -> high
    assert_eq!(c.len(), 5, "degree 4 => 5 coefficients");
    for (i, e) in expect.iter().enumerate() {
        assert_eq!(c[i], z(*e), "coeff of x^{}", i);
    }
}

#[test]
fn adv_rational_eigenvalues_known_spectrum() {
    // B = P * diag(1/2, -3, 2, 2) * P^{-1} with P integer (det 4); sympy gave
    // B = [[-15/4,-7/2,11/4,3/4],[-5/2,-1/2,5/2,0],[-13/4,-1,9/4,3/4],[-9,-9/2,3,7/2]]
    // and confirmed eigenvals {-3:1, 1/2:1, 2:2}.
    let data = vec![
        q(-15, 4), q(-7, 2), q(11, 4), q(3, 4),
        q(-5, 2), q(-1, 2), q(5, 2), q(0, 1),
        q(-13, 4), q(-1, 1), q(9, 4), q(3, 4),
        q(-9, 1), q(-9, 2), q(3, 1), q(7, 2),
    ];
    let b = Matrix::from_vec(4, 4, data).unwrap();
    let ev = b.rational_eigenvalues().unwrap();
    assert_eq!(ev.len(), 3, "three distinct rational eigenvalues: {:?}", ev);
    let total: usize = ev.iter().map(|(_, m)| m).sum();
    assert_eq!(total, 4, "full multiplicity 4: {:?}", ev);
    assert!(ev.contains(&(q(-3, 1), 1)), "missing -3: {:?}", ev);
    assert!(ev.contains(&(q(1, 2), 1)), "missing 1/2: {:?}", ev);
    assert!(ev.contains(&(q(2, 1), 2)), "missing 2 (mult 2): {:?}", ev);
}

#[test]
fn adv_lll_reduce_and_print() {
    // A basis of MY choosing (not the wiki example the implementers tested).
    let basis: Vec<Vec<Integer>> = vec![
        vec![z(105), z(821), z(404), z(328)],
        vec![z(881), z(667), z(644), z(927)],
        vec![z(181), z(483), z(87), z(500)],
        vec![z(893), z(834), z(732), z(441)],
    ];
    let (red, u) = lll_reduce_real::<BigFloat>(&basis, 3, 4, 128);

    // Exact integer check: red == u * basis (same lattice if u unimodular;
    // det(u) checked independently in python from the printout below).
    for i in 0..4 {
        for d in 0..4 {
            let mut s = z(0);
            for k in 0..4 {
                s = s + u[i][k].clone() * basis[k][d].clone();
            }
            assert_eq!(s, red[i][d], "red = u*basis fails at ({},{})", i, d);
        }
    }

    // Crate's own exact certificate (cross-checked independently in python).
    assert!(
        lll_is_reduced_exact(&red, 3, 4).unwrap(),
        "output not LLL-reduced per exact certificate"
    );
    assert!(
        !lll_is_reduced_exact(&basis, 3, 4).unwrap(),
        "input basis was already reduced — test would be vacuous"
    );

    // Print for the independent (python/fractions) verification.
    println!("REDUCED={:?}", red);
    println!("U={:?}", u);
}
