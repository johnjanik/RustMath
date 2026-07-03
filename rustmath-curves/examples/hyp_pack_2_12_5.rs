//! Does the hyperbolic (maximal) packing converge on the M24 [2,12,5] dessin where
//! the euclidean packing stalled (max angle error 8.5e-4, converged=false)?
//!
//! Run: cargo run --release -p rustmath-curves --example hyp_pack_2_12_5

use rustmath_curves::belyi::flag_packing::flag_pack;
use rustmath_curves::belyi::flags::flag_triangulation;
use rustmath_curves::belyi::hyperbolic_packing::{maximal_packing_of, HypPackingConfig};
use rustmath_curves::belyi::monodromy::Permutation;
use rustmath_curves::belyi::packing::{PackingComplex, PackingConfig};

const SIGMA0: [usize; 24] = [
    0, 14, 10, 9, 4, 5, 23, 17, 18, 3, 2, 11, 22, 13, 1, 15, 16, 7, 8, 19, 21, 20, 12, 6,
];
const SIGMA1: [usize; 24] = [
    14, 2, 22, 9, 16, 8, 13, 15, 18, 1, 23, 20, 3, 0, 21, 12, 19, 7, 17, 11, 10, 4, 5, 6,
];

fn main() {
    let s0 = Permutation::new(SIGMA0.to_vec()).unwrap();
    let s1 = Permutation::new(SIGMA1.to_vec()).unwrap();
    let tri = flag_triangulation(&s0, &s1).unwrap();
    let complex = PackingComplex::from_flags(&tri);

    println!("=== euclidean packing (baseline) ===");
    let el = flag_pack(&tri, &PackingConfig::default());
    println!(
        "  converged={}  iters={}  max_angle_err={:.3e}",
        el.packing.converged, el.packing.iterations, el.packing.max_angle_error
    );

    println!("\n=== hyperbolic maximal packing (barycentric subdivision) ===");
    let cfg = HypPackingConfig::default();
    let (sub, hp) = maximal_packing_of(&tri, &cfg);
    let min_flag_deg = (0..complex.n_vertices)
        .map(|v| complex.degree(v))
        .min()
        .unwrap();
    let min_sub_deg = (0..sub.n_vertices).map(|v| sub.degree(v)).min().unwrap();
    println!(
        "  flag complex: {} verts, min incident-tri = {}",
        complex.n_vertices, min_flag_deg
    );
    println!(
        "  subdivided:   {} verts, min incident-tri = {}",
        sub.n_vertices, min_sub_deg
    );
    println!(
        "  puncture vertex = {} (degree {})",
        hp.puncture,
        sub.degree(hp.puncture)
    );
    println!(
        "  converged={}  iters={}  max_angle_err={:.3e}",
        hp.converged, hp.iterations, hp.max_angle_error
    );

    // Radius spread over the interior: distinct, sensible radii ⇒ a healthy packing.
    let mut rmin = f64::INFINITY;
    let mut rmax = 0.0_f64;
    for v in 0..sub.n_vertices {
        if v == hp.puncture {
            continue;
        }
        let r = hp.radius(v);
        rmin = rmin.min(r);
        rmax = rmax.max(r);
    }
    println!("  interior hyperbolic radii: min={rmin:.6}  max={rmax:.6}");
    // How many interior vertices are actually badly off, and where?
    let mut errs: Vec<(f64, usize)> = (0..sub.n_vertices)
        .filter(|&v| v != hp.puncture)
        .map(|v| {
            let asum: f64 = sub
                .incident_of(v)
                .iter()
                .map(|&(a, b)| {
                    rustmath_curves::belyi::hyperbolic_packing::hyperbolic_flag_angle(
                        hp.s[v], hp.s[a], hp.s[b],
                    )
                })
                .sum();
            ((asum - 2.0 * std::f64::consts::PI).abs(), v)
        })
        .collect();
    errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let bad = errs.iter().filter(|(e, _)| *e > 1e-6).count();
    println!("  interior vertices with angle error > 1e-6: {bad}/{}", errs.len());
    println!(
        "  worst 3: {:?}",
        &errs[..3.min(errs.len())]
            .iter()
            .map(|(e, v)| (format!("{e:.2e}"), *v, format!("deg{}", sub.degree(*v))))
            .collect::<Vec<_>>()
    );

    println!(
        "\n  VERDICT: {}",
        if hp.converged {
            "hyperbolic packing CONVERGED — the sphere obstruction is gone; proceed to\n           \
             the hyperbolic layout to read off distinct, well-separated positions."
        } else {
            "did not converge — inspect the flower/incidence structure."
        }
    );
}
