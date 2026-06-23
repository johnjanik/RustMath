//! Class group vs gp bnfinit on small fields (index-1 defining polynomials).
use rustmath_integers::Integer;
use rustmath_numberfields::classgroup::class_group;
fn iz(v:&[i64])->Vec<Integer>{v.iter().map(|&x|Integer::from(x)).collect()}
#[test]
fn class_group_matches_gp() {
    assert_eq!(class_group(&iz(&[1, 0, 1])), Some(vec![]), "x^2+1");
    assert_eq!(class_group(&iz(&[5, 0, 1])), Some(vec![2]), "x^2+5");
    assert_eq!(class_group(&iz(&[14, 0, 1])), Some(vec![4]), "x^2+14");
    assert_eq!(class_group(&iz(&[21, 0, 1])), Some(vec![2, 2]), "x^2+21");
    assert_eq!(class_group(&iz(&[-10, 0, 1])), Some(vec![2]), "x^2-10");
    assert_eq!(class_group(&iz(&[6, 0, 1])), Some(vec![2]), "x^2+6");
    assert_eq!(class_group(&iz(&[6, -1, 1])), Some(vec![3]), "x2-x+6");
    assert_eq!(class_group(&iz(&[12, -1, 1])), Some(vec![5]), "x2-x+12");
    assert_eq!(class_group(&iz(&[-1, -1, 0, 1])), Some(vec![]), "x3-x-1");
}
