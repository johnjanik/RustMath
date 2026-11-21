//! Regenerate the Frobenius tables of the M23 curve X and of the six covers over the sextic
//! field L (WP-F, `docs/workstreams/m23_golay/HANDOFF.md` Section 3; MAGMA ch. 114
//! `LPolynomial`), with the status of every row.
//!
//! ```text
//! cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4
//! M23ISGALOIS=/path/to/m23isgalois cargo run --release -p rustmath-curves --example m23_zeta -- 5 127 4 --cover L
//! cargo run --release -p rustmath-curves --example m23_zeta -- 5 131 4 --json x.json
//! ```
//!
//! Options: `--cover L` (the covers over L at their degree-one primes; needs `M23ISGALOIS`),
//! `--model F` (X from HJLPPZ's F instead of the CC0 models; needs `M23ISGALOIS`),
//! `--reference-rule` (also print the counts of the reference rule of `curve.py` at `d05475b`,
//! every non-nodal point counted once: `reference rule N = [..]`, not certified), `--threads N`,
//! `--json <file>` (REQUESTS C-7: one object per row in the keys of the owner's `zeta_X.json`,
//! with the commit and the command; written before the first row with `"complete": false`,
//! rewritten after each row, and `"complete": true` at the end).
//! Without options X is counted on the CC0 models `[G, G7, G13]` (vendored, Le Lay).
//!
//! Each row prints its status: `Decided: nodal (62 nodes)` or
//! `Decided: delta-sum, Lemma 9.1 (..)` (Lemma 9.1 of the owner's paper, with the evidence of
//! absolute irreducibility and the singular points in the owner's labels), an `Unresolved`
//! status, or the refusal of every model. With `kmax >= 4` it also prints the order of Frobenius
//! on `Jac[2]` with its status and reason (for an open order, the rank bounds tried), the types
//! of `Jac[2]` and of the Golay Lagrangian `W` at the eigenvalue 1, and the cycle types of
//! Frobenius on `W` (8 letters through `GL_4(2) = A_8`, 15 points) (REQUESTS C-5, design C57).
//! The W columns rest on the hypothesis `H_W`, printed in a header line.

#[path = "../tests/common/m23.rs"]
mod m23;

use m23::*;
use rustmath_curves::nodal_biprojective::{normalization_point_counts, BranchRule, CountOptions};
use std::path::PathBuf;
use std::time::Instant;

/// The `--json` file of a run: rewritten after each row.
struct JsonOut {
    path: PathBuf,
    prov: Provenance,
    rows: Vec<serde_json::Value>,
}

impl JsonOut {
    fn write(&self, complete: bool) {
        let v = table_json(&self.prov, &self.rows, complete);
        let text = serde_json::to_string_pretty(&v).expect("JSON") + "\n";
        std::fs::write(&self.path, text)
            .unwrap_or_else(|e| panic!("cannot write {}: {e}", self.path.display()));
    }

    fn push(&mut self, row: &Row, secs: f64) {
        self.rows.push(row_json(row, Some(secs)));
        self.write(false);
    }
}

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
    let json_path = value("--json").map(PathBuf::from);
    let model_names: Vec<String> = if cover_l {
        vec!["calF".into()]
    } else if model_name.as_deref() == Some("F") {
        vec!["F".into()]
    } else {
        vec!["G".into(), "G7".into(), "G13".into()]
    };
    let external = (cover_l || model_name.as_deref() == Some("F")).then(require_m23isgalois);
    let mut json_out = json_path.map(|path| {
        let out = JsonOut {
            path,
            prov: Provenance::collect(
                if cover_l { "L" } else { "X" },
                model_names,
                (pmin, pmax, kmax),
                opts.threads,
                external.as_deref(),
            ),
            rows: Vec::new(),
        };
        out.write(false);
        out
    });
    if cover_l {
        let dir = require_m23isgalois();
        let calf = model_calf(&dir);
        println!(
            "# covers over L at degree-one primes (p, s - s0), calF from {}",
            dir.display()
        );
        if kmax as usize >= GENUS {
            println!("# the W columns assume H_W = {H_W}");
        }
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
                let secs = t.elapsed().as_secs_f64();
                println!("{} [{secs:.1} s]", line(&row, reference_rule));
                if let Some(j) = json_out.as_mut() {
                    j.push(&row, secs);
                }
            }
        }
    } else {
        let models = match model_name.as_deref() {
            Some("F") => vec![("F", model_f(&require_m23isgalois()))],
            _ => cc0_models(),
        };
        let names: Vec<&str> = models.iter().map(|m| m.0).collect();
        println!("# X, models {names:?}");
        if kmax as usize >= GENUS {
            println!("# the W columns assume H_W = {H_W}");
        }
        for p in pmin..=pmax {
            if p < 3 || !rustmath_integers::prime::is_prime_u64(p) {
                continue;
            }
            let t = Instant::now();
            let row = x_row(&models, p, kmax, &opts);
            let secs = t.elapsed().as_secs_f64();
            println!("{} [{secs:.1} s]", line(&row, reference_rule));
            if let Some(j) = json_out.as_mut() {
                j.push(&row, secs);
            }
        }
    }
    if let Some(j) = &json_out {
        j.write(true);
        println!("# wrote {}", j.path.display());
    }
    println!("# total {:.1} s", total.elapsed().as_secs_f64());
}
