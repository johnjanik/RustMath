//! THROWAWAY adversarial-verification tests — DELETE THIS FILE AFTER RUNNING.
//! Expected values derived independently with sympy before writing.

use rustmath_graphs::exact_spectra::{
    algebraic_connectivity_exact, graph_energy_exact, integer_spectrum, is_integral_graph,
    laplacian_integer_spectrum, spectral_radius_exact,
};
use rustmath_graphs::Graph;
use rustmath_integers::Integer;

fn z(n: i64) -> Integer {
    Integer::from(n)
}

fn complete(n: usize) -> Graph {
    let mut g = Graph::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            g.add_edge(i, j).unwrap();
        }
    }
    g
}

fn petersen() -> Graph {
    // Outer C5 on 0..4, inner pentagram 5..9 (i ~ i+2 mod 5), spokes i ~ i+5.
    let mut g = Graph::new(10);
    for i in 0..5 {
        g.add_edge(i, (i + 1) % 5).unwrap();
        g.add_edge(5 + i, 5 + ((i + 2) % 5)).unwrap();
        g.add_edge(i, i + 5).unwrap();
    }
    g
}

#[test]
fn adv_k6_spectrum() {
    // sympy: K6 adjacency spectrum {5:1, -1:5}
    let g = complete(6);
    assert!(is_integral_graph(&g));
    let spec = integer_spectrum(&g);
    assert_eq!(spec, vec![(z(-1), 5), (z(5), 1)], "K6 spectrum");
    assert_eq!(spectral_radius_exact(&g), Some(z(5)));
    assert_eq!(graph_energy_exact(&g), Some(z(10))); // 5 + 5*|-1|
}

#[test]
fn adv_petersen_spectrum() {
    // sympy (from my own edge list): adjacency {3:1, 1:5, -2:4},
    // Laplacian {0:1, 2:5, 5:4}.
    let g = petersen();
    assert!(is_integral_graph(&g));
    let spec = integer_spectrum(&g);
    assert_eq!(
        spec,
        vec![(z(-2), 4), (z(1), 5), (z(3), 1)],
        "Petersen adjacency spectrum"
    );
    let lspec = laplacian_integer_spectrum(&g);
    assert_eq!(
        lspec,
        vec![(z(0), 1), (z(2), 5), (z(5), 4)],
        "Petersen Laplacian spectrum"
    );
    assert_eq!(spectral_radius_exact(&g), Some(z(3)));
    assert_eq!(algebraic_connectivity_exact(&g), Some(z(2)));
    assert_eq!(graph_energy_exact(&g), Some(z(16))); // 3 + 5*1 + 4*2
}
