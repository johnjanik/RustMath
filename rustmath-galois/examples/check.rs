//! Manual validation harness for the Stauduhar descent (not a test; run with
//! `cargo run -p rustmath-galois --example check`). Prints the identified group
//! for a battery of known-answer polynomials.

use rustmath_galois::descent::{galois_group, Config};
use rustmath_integers::Integer;

fn iz(v: &[i64]) -> Vec<Integer> {
    v.iter().map(|&x| Integer::from(x)).collect()
}

fn main() {
    let cases: &[(&str, &[i64], &str)] = &[
        ("x^3-2", &[-2, 0, 0, 1], "3T2"),
        ("x^3-3x-1", &[-1, -3, 0, 1], "3T1"),
        ("x^4+1", &[1, 0, 0, 0, 1], "4T2"),
        ("x^4-2", &[-2, 0, 0, 0, 1], "4T3"),
        ("x^4+x+1", &[1, 1, 0, 0, 1], "4T5"),
        ("Phi5 (x^4+x^3+x^2+x+1)", &[1, 1, 1, 1, 1], "4T1"),
        ("x^4+8x+12", &[12, 8, 0, 0, 1], "4T4"),
        ("x^5-2", &[-2, 0, 0, 0, 0, 1], "5T3"),
        ("x^5-x-1", &[-1, -1, 0, 0, 0, 1], "5T5"),
        ("Q(z11)^+ ", &[1, 3, -3, -4, 1, 1], "5T1"),
        ("x^5-5x+12", &[12, -5, 0, 0, 0, 1], "5T2"),
    ];
    let cfg = Config::default();
    let mut pass = 0;
    for (name, coeffs, want) in cases {
        match galois_group(&iz(coeffs), &cfg) {
            Ok(r) => {
                let got = r.label.clone().unwrap_or_else(|| "?".into());
                let ok = got == *want;
                if ok {
                    pass += 1;
                }
                println!(
                    "{:>28}  want {:<5} got {:<5} order={:<3} steps={:?}  {}",
                    name,
                    want,
                    got,
                    r.order,
                    r.steps,
                    if ok { "OK" } else { "FAIL" }
                );
            }
            Err(e) => println!("{:>28}  ERROR: {e}", name),
        }
    }
    println!("\n{pass}/{} passed", cases.len());
}
