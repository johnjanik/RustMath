//! `rustmath-igp24` — a thin JSON CLI exposing the RustMath local-analysis engine
//! to the IGP24 pipeline (Sage/Python). Given a degree-`n` integer polynomial it
//! reports irreducibility, factorization over ℤ, discriminant (+ square test),
//! Frobenius cycle types and p-adic `(e,f)` ramification at a set of primes, and a
//! polred-reduced small-coefficient model.
//!
//! Usage:
//!   rustmath-igp24 "[a0,a1,...,an]"            # coeffs ascending (constant first)
//!   rustmath-igp24 "a0,a1,...,an" "5,7,11,13"  # explicit prime list
//!   echo "[...]" | rustmath-igp24              # coeffs on stdin
//!   rustmath-igp24 --batch --galois-fast       # one poly per stdin LINE, one
//!                                              # JSON result per stdout line
//!                                              # (atlas loaded once; for the
//!                                              #  OSCAR bridge / bulk narrowing)
//!
//! Output: a single JSON object on stdout (or one per line in --batch). Big
//! integers are emitted as JSON numbers (Python's json parses arbitrary precision).

use std::io::{BufRead, Read};
use std::str::FromStr;

use num_bigint::BigInt;
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::padic_factor::{cycle_type, ramification_type};
use rustmath_polynomials::{factor_over_integers, is_irreducible_over_integers, UnivariatePolynomial};
use rustmath_numberfields::polred::polred;
use rustmath_numberfields::round2::{field_discriminant, polredabs};
use rustmath_polynomials::resolvent::{
    galois_in_alternating, resolvent_orbit_signature, subset_sum_resolvent,
};
use rustmath_groups::transitive24::{separate_by_ksubset_orbits, CycleTypeSupport, Db};

/// First `n` primes (small sieve) for observing the Frobenius cycle-type support.
fn small_primes(n: usize) -> Vec<i64> {
    let mut ps = Vec::with_capacity(n);
    let mut x: i64 = 2;
    while ps.len() < n {
        if (2..).take_while(|d| d * d <= x).all(|d| x % d != 0) {
            ps.push(x);
        }
        x += 1;
    }
    ps
}

const DEFAULT_PRIMES: &[i64] = &[
    5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
];

fn parse_int_list(s: &str) -> Result<Vec<Integer>, String> {
    let t = s.trim().trim_start_matches('[').trim_end_matches(']');
    let mut out = Vec::new();
    for tok in t.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let b = BigInt::from_str(tok).map_err(|_| format!("bad integer: {tok:?}"))?;
        out.push(Integer::from(b));
    }
    if out.is_empty() {
        return Err("no coefficients".into());
    }
    Ok(out)
}

/// Exact integer square test by binary search (coefficients reach ~10^300, so
/// f64 is unusable; bigint binary search is ~log2(d) cheap multiplies).
fn is_perfect_square(d: &Integer) -> bool {
    if d.signum() < 0 {
        return false;
    }
    if d.is_zero() {
        return true;
    }
    let one = Integer::from(1i64);
    let two = Integer::from(2i64);
    let mut lo = Integer::from(0i64);
    let mut hi = d.clone();
    while lo <= hi {
        let mid = &(&lo + &hi) / &two;
        let sq = &mid * &mid;
        if &sq == d {
            return true;
        }
        if &sq < d {
            lo = &mid + &one;
        } else {
            hi = &mid - &one;
        }
    }
    false
}

fn coeffs_json(coeffs: &[Integer]) -> String {
    let parts: Vec<String> = coeffs.iter().map(|c| c.to_string()).collect();
    format!("[{}]", parts.join(","))
}

/// Opt-in heavy analyses (number-field reductions ~6–16s; galois identification).
struct Flags {
    polred: bool,
    polredabs: bool,
    fielddisc: bool,
    galois: bool,
    galois_fast: bool,
}

/// Full local analysis of one polynomial -> a single JSON object (no newline).
fn analyze(coeffs: &[Integer], primes: &[i64], fl: &Flags) -> String {
    let n = coeffs.len().saturating_sub(1);
    let poly = UnivariatePolynomial::new(coeffs.to_vec());

    // Irreducibility + factorization over ℤ.
    let irreducible = is_irreducible_over_integers(&poly).unwrap_or(false);
    let mut factors_json = Vec::new();
    if let Ok(facs) = factor_over_integers(&poly) {
        for (f, mult) in &facs {
            factors_json.push(format!(
                "{{\"factor\":{},\"mult\":{},\"degree\":{}}}",
                coeffs_json(f.coefficients()),
                mult,
                f.degree().unwrap_or(0)
            ));
        }
    }

    // Discriminant + square test (square => Galois ⊆ A_n).
    let disc = discriminant(coeffs);
    let disc_sq = is_perfect_square(&disc);

    // Frobenius cycle types and (e,f) ramification per prime.
    let mut cyc_json = Vec::new();
    let mut ram_json = Vec::new();
    for &p in primes {
        match cycle_type(coeffs, p) {
            Some(ct) => {
                let parts: Vec<String> = ct.iter().map(|x| x.to_string()).collect();
                cyc_json.push(format!("\"{p}\":[{}]", parts.join(",")));
            }
            None => cyc_json.push(format!("\"{p}\":null")),
        }
        match ramification_type(coeffs, p) {
            Ok(ef) => {
                let parts: Vec<String> = ef.iter().map(|(e, f)| format!("[{e},{f}]")).collect();
                ram_json.push(format!("\"{p}\":[{}]", parts.join(",")));
            }
            Err(_) => ram_json.push(format!("\"{p}\":null")),
        }
    }

    // Number-field reductions (only meaningful for irreducible input; gated).
    let polred_json = if fl.polred && irreducible {
        coeffs_json(&polred(coeffs))
    } else {
        "null".to_string()
    };
    let polredabs_json = if fl.polredabs && irreducible {
        coeffs_json(&polredabs(coeffs))
    } else {
        "null".to_string()
    };
    let fielddisc_json = if fl.fielddisc && irreducible {
        field_discriminant(coeffs).to_string()
    } else {
        "null".to_string()
    };

    // Native Galois identification: Frobenius support -> sound candidate class ->
    // k=2 resolvent orbit-signature narrowing (Stauduhar), no MAGMA / no slot.
    let galois_json = if (fl.galois || fl.galois_fast) && irreducible && n == 24 {
        let mut obs: Vec<Vec<usize>> = Vec::new();
        // Wider observation => tighter (still sound) candidate class: support
        // containment is monotone, the true group is never dropped.
        for p in small_primes(600) {
            if let Some(ct) = cycle_type(coeffs, p) {
                if !obs.contains(&ct) {
                    obs.push(ct);
                }
            }
        }
        let in_an = galois_in_alternating(coeffs);
        let cands = match CycleTypeSupport::load_default() {
            Ok(cts) => cts.candidates(&obs),
            Err(_) => Vec::new(),
        };
        let mut narrowed = cands.clone();
        let mut sig: Vec<usize> = Vec::new();
        if !fl.galois_fast && cands.len() > 1 {
            if let Ok(db) = Db::load_default() {
                let res = subset_sum_resolvent(coeffs, 2);
                if let Ok(s) = resolvent_orbit_signature(&res) {
                    sig = s.clone();
                    let sep = separate_by_ksubset_orbits(&db, &cands, 2, &s);
                    if !sep.is_empty() {
                        narrowed = sep;
                    }
                }
            }
        }
        let uniq = if narrowed.len() == 1 {
            narrowed[0].to_string()
        } else {
            "null".to_string()
        };
        format!(
            "{{\"candidate_class\":{:?},\"in_alternating\":{},\"k2_orbit_signature\":{:?},\
\"narrowed\":{:?},\"unique_t\":{}}}",
            cands, in_an, sig, narrowed, uniq
        )
    } else {
        "null".to_string()
    };

    format!(
        "{{\"degree\":{n},\"irreducible\":{irreducible},\"discriminant\":{disc},\
\"disc_is_square\":{disc_sq},\"factors\":[{}],\"cycle_types\":{{{}}},\
\"ramification\":{{{}}},\"polred\":{polred_json},\"polredabs\":{polredabs_json},\
\"field_discriminant\":{fielddisc_json},\"galois\":{galois_json}}}",
        factors_json.join(","),
        cyc_json.join(","),
        ram_json.join(",")
    )
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let batch = argv.iter().any(|a| a == "--batch");
    let fl = Flags {
        polred: argv.iter().any(|a| a == "--polred"),
        polredabs: argv.iter().any(|a| a == "--polredabs"),
        fielddisc: argv.iter().any(|a| a == "--fielddisc"),
        galois: argv.iter().any(|a| a == "--galois"),
        galois_fast: argv.iter().any(|a| a == "--galois-fast"),
    };
    // Positional (non-flag) args after the program name:
    //   single: [coeffs, primes?]   batch: [primes?]
    let nonflag: Vec<&String> = argv.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    let prime_src = if batch { nonflag.first() } else { nonflag.get(1) };
    let primes: Vec<i64> = match prime_src {
        Some(pa) => pa
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|t| t.trim().parse::<i64>().ok())
            .collect(),
        None => DEFAULT_PRIMES.to_vec(),
    };

    if batch {
        // One polynomial per stdin line, one JSON result per stdout line. The 24T
        // atlas (CycleTypeSupport / Db) is loaded fresh per line by analyze(); the
        // win over per-process invocation is amortizing process startup over the
        // whole stream, letting the OSCAR bridge narrow thousands of candidates.
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            match parse_int_list(t) {
                Ok(c) => println!("{}", analyze(&c, &primes, &fl)),
                Err(e) => println!("{{\"error\":\"{}\"}}", e.replace('"', "'")),
            }
        }
    } else {
        let coeff_src = if let Some(c) = nonflag.first() {
            (*c).clone()
        } else {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).ok();
            s
        };
        match parse_int_list(&coeff_src) {
            Ok(c) => println!("{}", analyze(&c, &primes, &fl)),
            Err(e) => {
                println!("{{\"error\":\"{}\"}}", e.replace('"', "'"));
                std::process::exit(1);
            }
        }
    }
}
