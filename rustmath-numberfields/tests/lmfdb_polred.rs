//! polred reduction on LMFDB degree-24 fields. Self-checks: monic degree-24,
//! squarefree, and not larger than the input (input is always a candidate).
//! Same-field correctness is guaranteed by construction (squarefree degree-24
//! charpoly of an element of Q(theta) is irreducible) and was confirmed
//! externally via gp nfisisom / polisirreducible on a 10-poly sample (2026-06-18).
use rustmath_integers::Integer;
use rustmath_numberfields::polred::polred;
use rustmath_polynomials::disc::discriminant;
#[rustfmt::skip]
const POLYS: &[[i64;25]] = &[
    [-87911, 440886, 3376455, -6216958, -19003989, 30843516, 34490860, -58691640, -27863199, 55412194, 10250409, -29163078, -943086, 9028542, -481767, -1686010, 173889, 189696, -25136, -12420, 1863, 430, -69, -6, 1],
    [82264637975788129, 0, -27744994428841640, 0, 4765639030479578, 0, -536739724201432, 0, 43522872227855, 0, -2646777443088, 0, 122651196780, 0, -4326183600, 0, 114512303, 0, -2211016, 0, 29594, 0, -248, 0, 1],
    [1827904000000, 0, 1827904000000, 0, 1188137600000, 0, 442915200000, 0, 118813760000, 0, 21266960000, 0, 2789176000, 0, 237952000, 0, 14466400, 0, 496600, 0, 11960, 0, 130, 0, 1],
];
fn supnorm(f:&[Integer])->Integer{ f.iter().map(|c|c.abs()).max().unwrap() }
#[test]
fn polred_degree24_valid_and_not_larger() {
    for (i,c) in POLYS.iter().enumerate() {
        let f:Vec<Integer>=c.iter().map(|&x|Integer::from(x)).collect();
        let r=polred(&f);
        assert_eq!(r.len(),25,"poly {} degree 24",i);
        assert_eq!(r[24],Integer::from(1),"monic");
        assert!(discriminant(&r)!=Integer::zero(),"squarefree (irreducible)");
        assert!(supnorm(&r)<=supnorm(&f),"not larger than input");
    }
}
