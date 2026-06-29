//! Müller M24 milestone harness.
//!
//! Reads `F_s(X) ∈ ℚ(t)[X]` (Müller's M24 family at a chosen `s`) in the
//! plain-text format emitted by `M23/extract_mueller_ff.py`, builds the
//! function field `ℚ(t)[X]/(F)` = the M23-fixed point-stabilizer cover, and runs
//! the Layer-A1 genus computation. The milestone (`g = 0`) is the end-to-end
//! validation of the whole stack against the result obtained by hand-descent.
//! A `NeedsHigherOrder(place)` outcome names the branch place whose φ-adic Montes
//! refinement is the remaining Layer-A1 build item.
//!
//!   cargo run -p rustmath-functionfields --example mueller_genus -- <datafile>

use rustmath_functionfields::function_field::ff_poly_from_coeffs;
use rustmath_functionfields::genus::{branch_radical, genus};
use rustmath_functionfields::genus_via_branch_cycles;
use rustmath_functionfields::ratfunc::{QtPoly, RationalFunction};
use rustmath_integers::Integer;
use rustmath_rationals::Rational;
use std::io::BufRead;

fn parse_rat(tok: &str) -> Rational {
    let (n, d) = tok.split_once('/').unwrap_or((tok, "1"));
    Rational::new(
        Integer::from_decimal_str(n.trim()).unwrap(),
        Integer::from_decimal_str(d.trim()).unwrap(),
    )
    .unwrap()
}

fn parse_qt(line: &str) -> QtPoly {
    let body = line.splitn(2, ':').nth(1).unwrap_or("").trim();
    let coeffs: Vec<Rational> = body.split_whitespace().map(parse_rat).collect();
    QtPoly::new(if coeffs.is_empty() { vec![Rational::from_i64(0)] } else { coeffs })
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: mueller_genus <datafile>");
    let file = std::fs::File::open(&path).expect("open datafile");
    let mut lines = std::io::BufReader::new(file).lines().map(|l| l.unwrap());

    let deg: usize = lines.next().unwrap().trim().parse().unwrap();
    let mut x_coeffs: Vec<RationalFunction> = Vec::with_capacity(deg + 1);
    for _ in 0..=deg {
        let num = parse_qt(&lines.next().unwrap());
        let den = parse_qt(&lines.next().unwrap());
        x_coeffs.push(RationalFunction::new(num, den).unwrap());
    }
    let f = ff_poly_from_coeffs(x_coeffs);
    println!("loaded F: deg_X = {:?}", f.degree());

    // Layer 6b: genus from branch cycles recovered mod p (fast — no resultant).
    println!("\n--- Layer 6b: genus via branch cycles recovered mod p ---");
    let primes = [101_i64, 103, 211, 307, 1009, 2003];
    let v = genus_via_branch_cycles(&f, &primes);
    match v.value {
        Some(g) => {
            println!("genus = {g}  [{:?}] {}", v.status, v.note);
            if g == 0 {
                println!("MILESTONE: g=0 recovered from F mod p — matches the hand-descent conic.");
            }
        }
        None => println!(
            "[{:?}] UNRESOLVED: {}\n(the F_p sweep misses branch points of residue degree > 1 — \
             the F_{{p^k}} extension sweep is the next build item.)",
            v.status, v.note
        ),
    }

    // Function-field route (Layer 2) as a cross-check — may be slow at degree 24.
    if std::env::args().any(|a| a == "--ff-genus") {
        println!("\n--- Layer 2 cross-check (function-field different) ---");
        println!("branch radical deg_t = {:?}", branch_radical(&f).degree());
        match genus(&f) {
            Ok(g) => println!("function-field genus = {g}"),
            Err(e) => println!("deferred: {e:?}"),
        }
    }
}
