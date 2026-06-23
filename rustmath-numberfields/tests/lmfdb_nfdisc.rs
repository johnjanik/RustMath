//! Field discriminant (Round 2) vs LMFDB `disc` column on degree-24 fields
//! (validated 2026-06-18). Slow (~16s/field); 3 fields committed, 5 checked locally.
use rustmath_integers::Integer;
use rustmath_numberfields::round2::field_discriminant;
#[rustfmt::skip]
const POLYS: &[[i64;25]] = &[
    [1, -6, 21, -58, 140, -280, 471, -696, 914, -1074, 1176, -1228, 1221, -1140, 966, -744, 537, -376, 265, -178, 107, -54, 21, -6, 1],
    [1, -6, 22, -61, 136, -248, 377, -454, 384, -108, -321, 721, -885, 721, -321, -108, 384, -454, 377, -248, 136, -61, 22, -6, 1],
    [1, 4, 9, 18, 38, 50, 34, -20, -46, 0, 53, 80, 39, -16, 9, -36, 47, -36, 35, -30, 16, -12, 5, -2, 1],
];
const DISCS: &[&str] = &[
    "368947264000000000000000000",
    "2341430542029089371659722001",
    "2754990144000000000000000000",
];
fn parse_abs(s:&str)->Integer{ let d=s.trim_start_matches('-'); let mut a=Integer::zero(); let t=Integer::from(10); for ch in d.bytes(){ a=a*t.clone()+Integer::from((ch-b'0') as i64);} a }
#[test]
fn field_disc_matches_lmfdb() {
    for (i,c) in POLYS.iter().enumerate() {
        let f:Vec<Integer>=c.iter().map(|&x|Integer::from(x)).collect();
        assert_eq!(field_discriminant(&f).abs(), parse_abs(DISCS[i]), "field disc poly {}", i);
    }
}
