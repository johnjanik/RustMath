//! Regenerate the Frobenius tables of the M23 curve X and of the six covers over the sextic
//! field L (WP-F, `docs/workstreams/m23_golay/HANDOFF.md` Section 3; MAGMA ch. 114
//! `LPolynomial`), with the status of every row.
//!
//! ```text
//! cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4
//! M23ISGALOIS=/path/to/m23isgalois cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4 --cover L
//! ```
//!
//! Options: `--cover L` (the covers over L at their degree-one primes; needs `M23ISGALOIS`),
//! `--model F` (X from HJLPPZ's F instead of the CC0 models; needs `M23ISGALOIS`),
//! `--reference-rule` (also print the counts of the reference rule of `curve.py` at `d05475b`,
//! every non-nodal point counted once: `reference rule N = [..]`, not certified), `--threads N`.
//! Without options X is counted on the CC0 models `[G, G7, G13]` (vendored, Le Lay).
//!
//! Each row prints its status: `Decided: nodal (62 nodes)` or
//! `Decided: delta-sum, Lemma 9.1 (..)` (Lemma 9.1 of the owner's paper, with the evidence of
//! absolute irreducibility and the singular points in the owner's labels), an `Unresolved`
//! status, or the refusal of every model.

#[path = "../tests/common/m23.rs"]
mod m23;

use m23::*;
use rustmath_curves::nodal_biprojective::{normalization_point_counts, BranchRule, CountOptions};
use std::time::Instant;

/// The row, with the reference-rule counts appended when asked.
fn line(row: &Row, reference_rule: bool) -> String {
    let mut s = format_row(row);
    if let (true, Ok(r)) = (reference_rule, &row.result) {
        if let Some(u) = r.counts_under(BranchRule::AssumedUnibranch) {
            s += &format!(" | reference rule N = {u:?}");
        }
    }
    s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| -> u64 { args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let (pmin, pmax, kmax) = (num(0, 5), num(1, 79), num(2, 4) as u32);
    let flag = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| {
        args.iter()
            .position(|a| a == f)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let cover_l = value("--cover").as_deref() == Some("L");
    let model_name = value("--model");
    let mut opts = CountOptions::default();
    let reference_rule = flag("--reference-rule");
    if let Some(t) = value("--threads").and_then(|s| s.parse().ok()) {
        opts.threads = Some(t);
    }
    let total = Instant::now();
    if cover_l {
        let dir = require_m23isgalois();
        let calf = model_calf(&dir);
        println!(
            "# covers over L at degree-one primes (p, s - s0), calF from {}",
            dir.display()
        );
        for p in pmin..=pmax {
            if p < 3 || !rustmath_integers::prime::is_prime_u64(p) {
                continue;
            }
            for s0 in degree_one_primes(p) {
                let t = Instant::now();
                let result = calf_mod(&calf, p, s0)
                    .and_then(|f| normalization_point_counts(&f, GENUS, kmax, &opts));
                assert_genus_consistent(&result, &format!("calF at p = {p}, s0 = {s0}"));
                let row = Row {
                    p,
                    s0: Some(s0),
                    model: "calF".into(),
                    result,
                };
                println!(
                    "{} [{:.1} s]",
                    line(&row, reference_rule),
                    t.elapsed().as_secs_f64()
                );
            }
        }
    } else {
        let models = match model_name.as_deref() {
            Some("F") => vec![("F", model_f(&require_m23isgalois()))],
            _ => cc0_models(),
        };
        let names: Vec<&str> = models.iter().map(|m| m.0).collect();
        println!("# X, models {names:?}");
        for p in pmin..=pmax {
            if p < 3 || !rustmath_integers::prime::is_prime_u64(p) {
                continue;
            }
            let t = Instant::now();
            let row = x_row(&models, p, kmax, &opts);
            println!(
                "{} [{:.1} s]",
                line(&row, reference_rule),
                t.elapsed().as_secs_f64()
            );
        }
    }
    println!("# total {:.1} s", total.elapsed().as_secs_f64());
}
