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
//!
//! Output: a single JSON object on stdout. Big integers are emitted as JSON
//! numbers (Python's json parses arbitrary precision).

use std::io::Read;
use std::str::FromStr;

use num_bigint::BigInt;
use rustmath_integers::Integer;
use rustmath_polynomials::disc::discriminant;
use rustmath_polynomials::padic_factor::{cycle_type, ramification_type};
use rustmath_polynomials::{factor_over_integers, is_irreducible_over_integers, UnivariatePolynomial};
use rustmath_numberfields::polred::polred;
use rustmath_numberfields::round2::{field_discriminant, polredabs};

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

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let coeff_src = if argv.len() >= 2 {
        argv[1].clone()
    } else {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).ok();
        s
    };
    let coeffs = match parse_int_list(&coeff_src) {
        Ok(c) => c,
        Err(e) => {
            println!("{{\"error\":\"{e}\"}}");
            std::process::exit(1);
        }
    };
    // The number-field reductions are expensive (polred ~6s, Round-2 polredabs /
    // field_discriminant ~16s), so they are opt-in; plain screening
    // (irreducibility / disc / cycle-types) needs none of them.
    //   --polred     equation-order small model
    //   --polredabs  maximal-order optimal model (smallest disc)
    //   --fielddisc  exact field discriminant (the IGP24-scored quantity)
    let want_polred = argv.iter().any(|a| a == "--polred");
    let want_polredabs = argv.iter().any(|a| a == "--polredabs");
    let want_fielddisc = argv.iter().any(|a| a == "--fielddisc");
    let prime_arg = argv.iter().skip(2).find(|a| !a.starts_with("--"));
    let primes: Vec<i64> = if let Some(pa) = prime_arg {
        pa
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .filter_map(|t| t.trim().parse::<i64>().ok())
            .collect()
    } else {
        DEFAULT_PRIMES.to_vec()
    };

    let n = coeffs.len().saturating_sub(1);
    let poly = UnivariatePolynomial::new(coeffs.clone());

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
    let disc = discriminant(&coeffs);
    let disc_sq = is_perfect_square(&disc);

    // Frobenius cycle types and (e,f) ramification per prime.
    let mut cyc_json = Vec::new();
    let mut ram_json = Vec::new();
    for &p in &primes {
        match cycle_type(&coeffs, p) {
            Some(ct) => {
                let parts: Vec<String> = ct.iter().map(|x| x.to_string()).collect();
                cyc_json.push(format!("\"{p}\":[{}]", parts.join(",")));
            }
            None => cyc_json.push(format!("\"{p}\":null")),
        }
        match ramification_type(&coeffs, p) {
            Ok(ef) => {
                let parts: Vec<String> =
                    ef.iter().map(|(e, f)| format!("[{e},{f}]")).collect();
                ram_json.push(format!("\"{p}\":[{}]", parts.join(",")));
            }
            Err(_) => ram_json.push(format!("\"{p}\":null")),
        }
    }

    // Number-field reductions (only meaningful for irreducible input; gated).
    let polred_json = if want_polred && irreducible {
        coeffs_json(&polred(&coeffs))
    } else {
        "null".to_string()
    };
    let polredabs_json = if want_polredabs && irreducible {
        coeffs_json(&polredabs(&coeffs))
    } else {
        "null".to_string()
    };
    let fielddisc_json = if want_fielddisc && irreducible {
        field_discriminant(&coeffs).to_string()
    } else {
        "null".to_string()
    };

    println!(
        "{{\"degree\":{n},\"irreducible\":{irreducible},\"discriminant\":{disc},\
\"disc_is_square\":{disc_sq},\"factors\":[{}],\"cycle_types\":{{{}}},\
\"ramification\":{{{}}},\"polred\":{polred_json},\"polredabs\":{polredabs_json},\
\"field_discriminant\":{fielddisc_json}}}",
        factors_json.join(","),
        cyc_json.join(","),
        ram_json.join(",")
    );
}
