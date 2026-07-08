//! belyi-atlas — deterministic, metadata-rich CLI over the frozen streamed KMSV
//! matrix assembler (`run_atlas_dump` -> `dump_scaled_ami_streamed`).
//!
//! The numerics are FROZEN: this binary only changes invocation, metadata, and
//! sidecars.  Every matrix entry is bit-identical to the trusted streamed harness
//! (`dump_2_12_5_matrix_ext_streamed`) for the same resolved parameters — both
//! paths funnel through `run_atlas_dump`.
//!
//!   belyi-atlas probe  --s0 .. --s1 .. --abc a,b,c --base i --center a [--coset j]
//!   belyi-atlas dump   --n N --s0 .. --s1 .. --abc a,b,c --base i --center a --out P
//!   belyi-atlas resume ...            (alias of dump; the streamed assembler
//!                                      auto-resumes from <out>.progress)
//!   belyi-atlas info   <matrix.bin>   (prints <matrix.bin>.meta.json)
//!
//! Output is one JSON event per line on stdout; human logs go to stderr.
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rug::Float;
use rustmath_curves::belyi::modular_forms_hp::{run_atlas_dump, AtlasDumpParams};

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn parse_perm(s: &str) -> Vec<usize> {
    s.split(',').map(|t| t.trim().parse().expect("permutation entry")).collect()
}

/// Series length N to reach `floor` accuracy for a chart of radius `rho`,
/// rounded up to the next multiple of 50.  -ln(1e-14) = 32.2361913...
fn n_for_floor(rho: f64, floor: f64) -> usize {
    let raw = (-floor.ln()) / (-rho.ln());
    ((raw / 50.0).ceil() as usize) * 50
}

/// Collect `--key value` pairs; bare `--flag` becomes `flag=true`.
fn flags(args: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                m.insert(key.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                m.insert(key.to_string(), "true".to_string());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    m
}

fn params_from_flags(f: &HashMap<String, String>, default_n: usize, default_limbs: usize) -> AtlasDumpParams {
    let g = |k: &str| f.get(k).cloned();
    let abc: Vec<u32> = g("abc")
        .expect("--abc a,b,c required")
        .split(',')
        .map(|x| x.trim().parse().expect("abc entry"))
        .collect();
    assert_eq!(abc.len(), 3, "--abc must be a,b,c");
    AtlasDumpParams {
        s0: parse_perm(&g("s0").expect("--s0 required")),
        s1: parse_perm(&g("s1").expect("--s1 required")),
        abc: (abc[0], abc[1], abc[2]),
        base: g("base").and_then(|s| s.parse().ok()).unwrap_or(0),
        n: g("n").and_then(|s| s.parse().ok()).unwrap_or(default_n),
        k: g("k").and_then(|s| s.parse().ok()).unwrap_or(4),
        prec: g("prec").and_then(|s| s.parse().ok()).unwrap_or(140),
        nlimbs: g("limbs").and_then(|s| s.parse().ok()).unwrap_or(default_limbs),
        r_prune: g("rprune").and_then(|s| s.parse().ok()).unwrap_or(0.99995),
        l_max: g("lmax").and_then(|s| s.parse().ok()).unwrap_or(100),
        center: g("center").unwrap_or_else(|| "a".into()),
        coset: g("coset").and_then(|s| s.parse().ok()).unwrap_or(1),
        out: g("out").unwrap_or_else(|| "/tmp/belyi_atlas.bin".into()),
    }
}

fn perm_json(v: &[usize]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

/// Write `<out>.meta.json` atomically (tmp then rename).  Downstream consumers
/// read this, never stdout logs.
fn write_meta(p: &AtlasDumpParams, rho: &Float, ctr_re: &str, ctr_im: &str, started: u64, finished: u64) {
    let (a, b, c) = p.abc;
    let meta = format!(
        "{{\n\
         \x20 \"schema\": 1,\n\
         \x20 \"out\": \"{out}\",\n\
         \x20 \"dim\": {dim},\n\
         \x20 \"n\": {n},\n\
         \x20 \"k\": {k},\n\
         \x20 \"prec\": {prec},\n\
         \x20 \"limbs\": {limbs},\n\
         \x20 \"base\": {base},\n\
         \x20 \"center\": \"{center}\",\n\
         \x20 \"coset\": {coset},\n\
         \x20 \"abc\": [{a}, {b}, {c}],\n\
         \x20 \"rprune\": {rprune},\n\
         \x20 \"lmax\": {lmax},\n\
         \x20 \"q\": {q},\n\
         \x20 \"rho\": \"{rho}\",\n\
         \x20 \"ctr\": [\"{ctr_re}\", \"{ctr_im}\"],\n\
         \x20 \"s0\": {s0},\n\
         \x20 \"s1\": {s1},\n\
         \x20 \"started_unix\": {started},\n\
         \x20 \"finished_unix\": {finished}\n\
         }}\n",
        out = p.out, dim = p.n + 1, n = p.n, k = p.k, prec = p.prec, limbs = p.nlimbs,
        base = p.base, center = p.center, coset = p.coset, a = a, b = b, c = c,
        rprune = p.r_prune, lmax = p.l_max, q = 2 * p.n + 8,
        rho = format!("{:.30}", rho), ctr_re = ctr_re, ctr_im = ctr_im,
        s0 = perm_json(&p.s0), s1 = perm_json(&p.s1), started = started, finished = finished,
    );
    let path = format!("{}.meta.json", p.out);
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, meta).expect("write meta tmp");
    std::fs::rename(&tmp, &path).expect("rename meta");
}

fn run_and_report(p: &AtlasDumpParams, is_probe: bool) {
    eprintln!(
        "[belyi-atlas] {} center={} N={} limbs={} prec={} base={} -> {}",
        if is_probe { "probe" } else { "dump" }, p.center, p.n, p.nlimbs, p.prec, p.base, p.out
    );
    let started = now_unix();
    let r = run_atlas_dump(p);
    let finished = now_unix();
    let ctr_re = format!("{:.40}", r.ctr.real());
    let ctr_im = format!("{:.40}", r.ctr.imag());
    write_meta(p, &r.rho, &ctr_re, &ctr_im, started, finished);
    if is_probe {
        let auto_n = n_for_floor(r.rho.to_f64(), 1e-14);
        println!(
            "{{\"event\":\"probe_result\",\"rho\":\"{:.20}\",\"ctr\":[\"{}\",\"{}\"],\"n_auto\":{},\"meta\":\"{}.meta.json\"}}",
            r.rho, ctr_re, ctr_im, auto_n, p.out
        );
    } else {
        println!(
            "{{\"event\":\"done\",\"path\":\"{}\",\"rows\":{},\"rho\":\"{:.20}\",\"meta\":\"{}.meta.json\"}}",
            p.out, r.dim, r.rho, p.out
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: belyi-atlas <probe|dump|resume|info> [--flags]");
        std::process::exit(2);
    }
    match args[1].as_str() {
        "probe" => run_and_report(&params_from_flags(&flags(&args[2..]), 80, 1), true),
        "dump" | "resume" => run_and_report(&params_from_flags(&flags(&args[2..]), 600, 2), false),
        "info" => {
            let path = args
                .get(2)
                .filter(|s| !s.starts_with("--"))
                .cloned()
                .expect("info <matrix.bin>");
            let meta = format!("{}.meta.json", path);
            match std::fs::read_to_string(&meta) {
                Ok(s) => print!("{}", s),
                Err(e) => {
                    eprintln!("no meta sidecar {}: {}", meta, e);
                    std::process::exit(1);
                }
            }
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            std::process::exit(2);
        }
    }
}
