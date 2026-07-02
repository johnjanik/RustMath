//! Audit the genuine M24 [2,12,5] flag triangulation and test each regularizer
//! against the valence gate — letting data decide the leaf-degeneracy fix.
//!
//! Run: cargo run -p rustmath-curves --example audit_2_12_5

use rustmath_curves::belyi::flags::flag_triangulation_of;
use rustmath_curves::belyi::monodromy::{BelyiTriple, Permutation};
use rustmath_curves::belyi::regularize::{regularize_for_circle_packing, RegularizationMethod};
use rustmath_curves::belyi::triangulation_audit::Triangulation;

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];
const SIGMAINF: [usize; 24] = [
    13, 0, 20, 3, 21, 22, 10, 18, 8, 12, 1, 19, 2, 6, 9, 7, 4, 17, 5, 16, 14, 11, 15, 23,
];

fn main() {
    let t = BelyiTriple {
        sigma0: Permutation::new(SIGMA0.to_vec()).unwrap(),
        sigma1: Permutation::new(SIGMA1.to_vec()).unwrap(),
        sigmainf: Permutation::new(SIGMAINF.to_vec()).unwrap(),
    };
    let ft = flag_triangulation_of(&t).unwrap();
    let tri = Triangulation::from_flags(&ft);

    println!("=== raw 4E flag triangulation of the M24 [2,12,5] dessin ===");
    let a = tri.audit();
    println!("  vertices={} edges={} faces={}  χ={}", a.n_vertices, a.n_edges, a.n_faces, a.euler_characteristic);
    println!("  min_valence={}  #degree-2={}  #isolated={}  #nonmanifold_edges={}",
        a.min_valence, a.degree_two_vertices.len(), a.isolated_vertices.len(), a.nonmanifold_edges.len());
    println!("  degree-2 vertices: {:?}", a.degree_two_vertices);
    println!("  sphere_candidate={}  packable={}", tri.is_sphere_candidate(), tri.is_packable_by_valence_gate());
    println!();

    for m in [
        RegularizationMethod::None,
        RegularizationMethod::FullSubdivision,
        RegularizationMethod::LocalLeafCap,
    ] {
        let (out, r) = regularize_for_circle_packing(&tri, m);
        println!("=== {:?} ===", m);
        println!("  min_valence {} -> {}", r.before_min_valence, r.after_min_valence);
        println!("  degree-2   {} -> {}", r.before_degree_two, r.after_degree_two);
        println!("  χ          {} -> {}", r.euler_before, r.euler_after);
        let oa = out.audit();
        println!("  out: vertices={} faces={} nonmanifold_edges={}", oa.n_vertices, oa.n_faces, oa.nonmanifold_edges.len());
        println!("  >>> PACKABLE (valence gate): {}", r.packable);
        println!();
    }
}
