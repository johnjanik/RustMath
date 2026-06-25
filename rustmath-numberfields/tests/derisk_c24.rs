use rustmath_numberfields::abext::gaussian_period_poly;
#[test] #[ignore]
fn build_c24_degree24() {
    // Degree-24 subfield of Q(zeta_73): 73 = 1 mod 24, cyclic C24, conductor 73.
    let poly = gaussian_period_poly(73, 24).expect("gaussian period poly deg 24");
    assert_eq!(poly.len(), 25, "must be degree 24 (25 coeffs)");
    // emit ascending coeffs as a JSON list for piping into rustmath-igp24 --galois-short
    let s: Vec<String> = poly.iter().map(|c| c.to_string()).collect();
    eprintln!("C24_COEFFS=[{}]", s.join(","));
}
