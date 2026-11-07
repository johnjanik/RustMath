//! `Parent::is_finite` / `cardinality` for matrix algebras and lattices (MAGMA Handbook
//! Ch. 17 §17.4.3 `IsFinite`: true if known finite, false if known infinite, error, here
//! `None`, if unknown; Ch. 83 matrix algebras; Ch. 30 lattices).

use rustmath_core::Parent;
use rustmath_integers::Integer;
use rustmath_matrix::{Lattice, MatrixAlgebra};
use rustmath_rationals::Rational;

fn z(n: i64) -> Integer {
    Integer::from(n)
}

// GAP: IsFinite(FullMatrixAlgebra(Rationals, 2)) -> false.
#[test]
fn matrix_algebra_over_q_is_infinite() {
    let m = MatrixAlgebra::<Rational>::full(2);
    assert_eq!(m.is_finite(), Some(false));
    assert_eq!(m.cardinality(), None);
}

// M_0(F) is the zero ring: one element.
#[test]
fn zero_matrix_algebra_is_finite() {
    let m = MatrixAlgebra::<Rational>::full(0);
    assert_eq!(m.is_finite(), Some(true));
    assert_eq!(m.cardinality(), Some(1));
}

// The Z-span of a nonzero vector is infinite.
#[test]
fn lattice_finiteness() {
    let l = Lattice::with_basis(vec![vec![z(1), z(0)], vec![z(0), z(1)]]).unwrap();
    assert_eq!(l.is_finite(), Some(false));
    assert_eq!(l.cardinality(), None);
    // A zero row next to a nonzero one does not make it finite.
    let mixed = Lattice::with_basis(vec![vec![z(0), z(0)], vec![z(1), z(2)]]).unwrap();
    assert_eq!(mixed.is_finite(), Some(false));
    // The lattice spanned by the zero vector is {0}.
    let zero = Lattice::with_basis(vec![vec![z(0), z(0)]]).unwrap();
    assert_eq!(zero.is_finite(), Some(true));
    assert_eq!(zero.cardinality(), Some(1));
}
