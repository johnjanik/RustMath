//! Run a *genuine* M₂₄ `[2,12,5]` dessin through the 4E flag triangulation and
//! confirm it is a valid genus-0 GEM with the passport's counts.
//!
//! The triple was computed by `inverse_galois/M23/find_triple_2_12_5.sage`
//! (`libgap.MathieuGroup(24)`, random conjugates of the 2A and 12² class reps,
//! filtered to product-type 5⁴1⁴ + ⟨g1,g2⟩ = M₂₄, exported in RustMath's
//! function-composition convention σ0∘σ1∘σ∞ = id). Found after 846 samples;
//! ⟨g1,g2⟩ has order 244_823_040 = |M₂₄|.

use rustmath_curves::belyi::flags::flag_triangulation_of;
use rustmath_curves::belyi::monodromy::{BelyiTriple, Permutation};

// σ0: 2^8 1^8, σ1: 12^2, σ∞: 5^4 1^4, with σ0∘σ1∘σ∞ = id.
const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];
const SIGMAINF: [usize; 24] = [
    13, 0, 20, 3, 21, 22, 10, 18, 8, 12, 1, 19, 2, 6, 9, 7, 4, 17, 5, 16, 14, 11, 15, 23,
];

fn triple() -> BelyiTriple {
    BelyiTriple {
        sigma0: Permutation::new(SIGMA0.to_vec()).unwrap(),
        sigma1: Permutation::new(SIGMA1.to_vec()).unwrap(),
        sigmainf: Permutation::new(SIGMAINF.to_vec()).unwrap(),
    }
}

#[test]
fn triple_is_a_valid_2_12_5_dessin() {
    let t = triple();
    // σ0∘σ1∘σ∞ = id AND ⟨σ0,σ1⟩ transitive (the dessin condition).
    t.validate().expect("valid transitive dessin with product identity");

    assert_eq!(
        t.sigma0.cycle_lengths(),
        vec![2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        "σ0 is 2^8 1^8"
    );
    assert_eq!(t.sigma1.cycle_lengths(), vec![12, 12], "σ1 is 12^2");
    assert_eq!(
        t.sigmainf.cycle_lengths(),
        vec![5, 5, 5, 5, 1, 1, 1, 1],
        "σ∞ is 5^4 1^4"
    );

    // Riemann–Hurwitz genus straight from the branch cycles.
    assert_eq!(t.genus().unwrap(), 0, "the [2,12,5] passport has genus 0");
}

#[test]
fn flag_triangulation_of_real_dessin_is_genus_zero() {
    let t = triple();
    let tri = flag_triangulation_of(&t).expect("flag triangulation");

    // A valid graph-encoded map whose flag orbits reproduce this dessin.
    assert!(tri.is_valid_gem(), "valid GEM");
    assert!(tri.orbits_match_dessin(), "flag orbits match the dessin");

    // Passport counts: 2^8 1^8 ⇒ 16 black, 12^2 ⇒ 2 white, 5^4 1^4 ⇒ 8 faces.
    assert_eq!(tri.n_black, 16, "16 black vertices");
    assert_eq!(tri.n_white, 2, "2 white vertices");
    assert_eq!(tri.n_face, 8, "8 faces");
    assert_eq!(tri.degree, 24, "degree 24");
    assert_eq!(tri.n_triangles(), 96, "4E = 96 flags");

    // The headline: genus 0 on the real M₂₄ dessin.
    assert_eq!(tri.euler_characteristic(), 2, "χ = 2");
    assert_eq!(tri.genus(), 0, "genus 0");

    // Euler cross-check V − E + T with V = 16+2+8+24 = 50, T = 96 ⇒ E = 144.
    assert_eq!(tri.n_vertices(), 50);
}
